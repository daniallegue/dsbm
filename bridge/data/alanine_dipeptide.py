import hashlib
import os

import numpy as np
import torch

from ..sde.manifold import SO3

SMILES = "[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH3:6])[C:7](=[O:8])[NH:9][CH3:10]"

_HEAVY_NAMES = {
    1: ("CH3", "ACE"), 2: ("C", "ACE"), 3: ("O", "ACE"),
    4: ("N", "ALA"), 5: ("CA", "ALA"), 6: ("CB", "ALA"), 7: ("C", "ALA"), 8: ("O", "ALA"),
    9: ("N", "NME"), 10: ("CH3", "NME"),
}
_H_NAMES = {
    "CH3": ["HH31", "HH32", "HH33"],
    "N": ["H"],
    "CA": ["HA"],
    "CB": ["HB1", "HB2", "HB3"],
    "C": [], "O": [],
}

# (phi, psi) basin centers in degrees, from the classical alanine dipeptide PMF
BASIN_CENTERS_DEG = {
    "c7eq": (-83.0, 75.0),
    "c7ax": (70.0, -70.0),
}


def _wrap(angle):
    """Wrap angle(s) in radians to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def dihedral_from_bonds(b0, b1, b2):
    """Dihedral angle (radians) from three consecutive bond vectors (direction only,
    magnitude-invariant), batched over leading dims. b_i: (..., 3)."""
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)
    return np.arctan2(y, x)


def dihedral_angle(p0, p1, p2, p3):
    """Dihedral angle (radians) of four points, batched over leading dims. p_i: (..., 3)."""
    return dihedral_from_bonds(p0 - p1, p2 - p1, p3 - p2)


def angle_between(v1, v2):
    """Angle (radians) between two vectors, batched over leading dims. v_i: (..., 3)."""
    v1 = _normalize(v1)
    v2 = _normalize(v2)
    cos_t = (v1 * v2).sum(-1).clip(-1.0, 1.0)
    return np.arccos(cos_t)


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def backbone_frame(x1, x2, x3):
    """
    Right-handed orthonormal frame anchored at x2, built from (x1, x2, x3) via
    Gram-Schmidt (AlphaFold/FrameDiff convention). 
    
    Returns rotation matrix (..., 3, 3) with columns (e1, e2, e3).
    """
    e1 = _normalize(x1 - x2)
    u2 = x3 - x2
    e2 = _normalize(u2 - (e1 * u2).sum(-1, keepdims=True) * e1)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=-1)


def _build_conformer(phi_deg, psi_deg, seed=0):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms
    import openmm.app as app

    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

    idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}

    conf = mol.GetConformer()
    # Rotating about a bond only moves the fragment downstream of it, so setting
    # these after the free UFF relaxation cannot reintroduce steric clashes elsewhere.
    rdMolTransforms.SetDihedralDeg(conf, idx[1], idx[2], idx[4], idx[5], 180.0)
    rdMolTransforms.SetDihedralDeg(conf, idx[2], idx[4], idx[5], idx[7], float(phi_deg))
    rdMolTransforms.SetDihedralDeg(conf, idx[4], idx[5], idx[7], idx[9], float(psi_deg))
    rdMolTransforms.SetDihedralDeg(conf, idx[5], idx[7], idx[9], idx[10], 180.0)

    name_res = {}
    for amap, (name, res) in _HEAVY_NAMES.items():
        heavy_idx = idx[amap]
        name_res[heavy_idx] = (name, res)
        h_names = _H_NAMES[name]
        h_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(heavy_idx).GetNeighbors() if n.GetSymbol() == "H"]
        for h_idx, h_name in zip(h_neighbors, h_names):
            name_res[h_idx] = (h_name, res)

    # openmm.app.Topology requires all atoms of a residue to be added contiguously,
    # but RDKit's post-AddHs indexing interleaves residues (heavy atoms first, then
    # hydrogens appended at the end)
    ordered_idx = []
    for res_name in ("ACE", "ALA", "NME"):
        ordered_idx += [i for i in range(mol.GetNumAtoms()) if name_res[i][1] == res_name]

    top = app.Topology()
    chain = top.addChain()
    res_objs = {res_name: top.addResidue(res_name, chain) for res_name in ("ACE", "ALA", "NME")}

    atom_objs = {}
    for rdkit_idx in ordered_idx:
        name, res_name = name_res[rdkit_idx]
        element = app.element.get_by_symbol(mol.GetAtomWithIdx(rdkit_idx).GetSymbol())
        atom_objs[rdkit_idx] = top.addAtom(name, element, res_objs[res_name])

    for bond in mol.GetBonds():
        top.addBond(atom_objs[bond.GetBeginAtomIdx()], atom_objs[bond.GetEndAtomIdx()])

    positions_ang = conf.GetPositions()[ordered_idx]  # reorder to match addAtom order
    positions_nm = positions_ang * 0.1
    return top, positions_nm


def _run_langevin(topology, positions_nm, temperature_k, friction_per_ps, n_steps, save_every, seed=0):
    """Vacuum AMBER99SB Langevin trajectory. Returns positions array (n_saved, n_atoms, 3) in nm."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    forcefield = app.ForceField("amber99sb.xml")
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(temperature_k * unit.kelvin, friction_per_ps / unit.picosecond,
                                              1.0 * unit.femtosecond)
    integrator.setRandomNumberSeed(seed)
    simulation = app.Simulation(topology, system, integrator, mm.Platform.getPlatformByName("CPU"))
    simulation.context.setPositions(positions_nm * unit.nanometer)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)

    n_atoms = topology.getNumAtoms()
    n_saved = n_steps // save_every
    traj = np.empty((n_saved, n_atoms, 3), dtype=np.float64)
    for k in range(n_saved):
        simulation.step(save_every)
        state = simulation.context.getState(getPositions=True)
        traj[k] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return traj


def _atom_indices(topology):
    """
    Look up the 5 backbone atoms needed for (phi, psi) and the two frames, by
    (residue, name) rather than a hardcoded position
    """
    lookup = {(atom.residue.name, atom.name): atom.index for atom in topology.atoms()}
    return dict(ace_c=lookup[("ACE", "C")], ala_n=lookup[("ALA", "N")], ala_ca=lookup[("ALA", "CA")],
                ala_c=lookup[("ALA", "C")], nme_n=lookup[("NME", "N")])


def _phi_psi(traj, idx):
    p = traj
    phi = dihedral_angle(p[:, idx["ace_c"]], p[:, idx["ala_n"]], p[:, idx["ala_ca"]], p[:, idx["ala_c"]])
    psi = dihedral_angle(p[:, idx["ala_n"]], p[:, idx["ala_ca"]], p[:, idx["ala_c"]], p[:, idx["nme_n"]])
    return phi, psi


def _frames(traj, idx):
    """
    (n_frames, 22, 3) positions -> (n_frames, 3, 3) relative rotation R1^T @ R2
    """
    p = traj
    r1 = backbone_frame(p[:, idx["ace_c"]], p[:, idx["ala_n"]], p[:, idx["ala_ca"]])
    r2 = backbone_frame(p[:, idx["ala_ca"]], p[:, idx["ala_c"]], p[:, idx["nme_n"]])
    return np.matmul(r1.transpose(0, 2, 1), r2)


def calibrate_bond_angles(seed=0):
    """(phi, psi) are recoverable from the two backbone frames alone, up to two fixed
    local bond angles that the frame construction does not itself encode:
      theta_n = angle(ACE.C - N, ALA.CA - N)        at the N vertex (frame 1's anchor)
      theta_c = angle(ALA.CA - ALA.C, NME.N - ALA.C) at the C vertex (frame 2's anchor)
    """
    topology, positions = _build_conformer(*BASIN_CENTERS_DEG["c7eq"], seed=seed)
    idx = _atom_indices(topology)

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    forcefield = app.ForceField("amber99sb.xml")
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1.0 * unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator, mm.Platform.getPlatformByName("CPU"))
    simulation.context.setPositions(positions * unit.nanometer)
    simulation.minimizeEnergy()
    p = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    theta_n = angle_between(p[idx["ace_c"]] - p[idx["ala_n"]], p[idx["ala_ca"]] - p[idx["ala_n"]])
    theta_c = angle_between(p[idx["ala_ca"]] - p[idx["ala_c"]], p[idx["nme_n"]] - p[idx["ala_c"]])
    return float(theta_n), float(theta_c)


def frames_to_phi_psi(frames, theta_n, theta_c):
    """
    Reconstruct (phi, psi) in radians directly from the relative backbone rotation
    R1^T @ R2 (n, 3, 3), using the fixed local bond angles from calibrate_bond_angles.
    """
    e1_n = np.array([1.0, 0.0, 0.0])
    e2_n = np.array([0.0, 1.0, 0.0])
    e1_c, e2_c = frames[:, :, 0], frames[:, :, 1]  # frame 2's basis, in frame 1's local coordinates

    ca_minus_n = np.cos(theta_n) * e1_n + np.sin(theta_n) * e2_n     # ALA.CA - N direction
    nmen_minus_c = np.cos(theta_c) * e1_c + np.sin(theta_c) * e2_c   # NME.N - ALA.C direction

    phi = dihedral_from_bonds(np.broadcast_to(e1_n, e1_c.shape), np.broadcast_to(ca_minus_n, e1_c.shape), -e1_c)
    psi = dihedral_from_bonds(-np.broadcast_to(ca_minus_n, e1_c.shape), -e1_c, nmen_minus_c)
    return phi, psi


def generate_basin_frames(basin, n_samples, temperature_k=300.0, friction_per_ps=1.0, save_every=10,
                           ball_radius=0.75, seed=0, max_chunks=50, verbose=True):
    """
    Unbiased Langevin dynamics seeded in `basin`, keeping only configurations within `ball_radius` (radians, wrapped)
    
    Returns (n_samples, 3, 3) relative rotation matrices."""
    phi0_deg, psi0_deg = BASIN_CENTERS_DEG[basin]
    phi0, psi0 = np.deg2rad(phi0_deg), np.deg2rad(psi0_deg)

    topology, positions = _build_conformer(phi0_deg, psi0_deg, seed=seed)
    idx = _atom_indices(topology)

    collected = []
    n_collected = 0
    chunk_steps = max(save_every * 200, 20000)
    for chunk in range(max_chunks):
        traj = _run_langevin(topology, positions, temperature_k, friction_per_ps, chunk_steps, save_every,
                              seed=seed * 1000 + chunk)
        positions = traj[-1]  # continue from where this chunk left off
        phi, psi = _phi_psi(traj, idx)
        dist = np.sqrt(_wrap(phi - phi0) ** 2 + _wrap(psi - psi0) ** 2)
        mask = dist < ball_radius
        kept = _frames(traj[mask], idx)
        collected.append(kept)
        n_collected += kept.shape[0]
        if verbose:
            print(f"[alanine_dipeptide:{basin}] chunk {chunk}: {mask.sum()}/{len(mask)} in-basin, "
                  f"total {n_collected}/{n_samples}")
        if n_collected >= n_samples:
            break
    else:
        raise RuntimeError(f"basin '{basin}' did not stay confined to its (phi,psi) ball over "
                            f"{max_chunks} chunks -- likely crossed the barrier or ball_radius is too tight")

    frames = np.concatenate(collected, axis=0)[:n_samples]
    return frames


def frames_to_flat_tensor(frames):
    """(n, 3, 3) relative rotation matrices -> (n, 9) tensor matching SO3(n_copies=1)'s
    flattened ambient representation."""
    return torch.from_numpy(frames.reshape(frames.shape[0], 9)).float()


class AlanineDipeptideFramesDataset(torch.utils.data.Dataset):
    """Empirical relative-backbone-frame (single SO(3), R1^T @ R2) marginal for one
    metastable basin ('c7eq' or 'c7ax') of alanine dipeptide. y is an unused
    placeholder."""

    def __init__(self, basin, n_samples, cache_dir=None, temperature_k=300.0, friction_per_ps=1.0,
                 save_every=10, ball_radius=0.75, seed=0):
        assert basin in BASIN_CENTERS_DEG, f"unknown basin '{basin}', expected one of {list(BASIN_CENTERS_DEG)}"

        # "relframe_v2" versions the cache: v1 stored the two absolute frames (R1, R2),
        # which turned out to be dominated by the molecule's unrestrained lab-frame
        # tumbling in vacuum MD (see _frames' docstring) -- bumping this avoids silently
        # loading a stale (n, 18) v1 cache file as if it were the new (n, 9) format.
        cache_key = (basin, n_samples, temperature_k, friction_per_ps, save_every, ball_radius, seed, "relframe_v2")
        cache_hash = hashlib.sha1(str(cache_key).encode()).hexdigest()[:12]
        cache_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{basin}_{cache_hash}.npy")

        if cache_path is not None and os.path.isfile(cache_path):
            frames = np.load(cache_path)
        else:
            frames = generate_basin_frames(basin, n_samples, temperature_k=temperature_k,
                                            friction_per_ps=friction_per_ps, save_every=save_every,
                                            ball_radius=ball_radius, seed=seed)
            if cache_path is not None:
                np.save(cache_path, frames)

        self.points = frames_to_flat_tensor(frames)
        self.manifold = SO3(n_copies=1)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        return self.points[index], torch.zeros((1,))
