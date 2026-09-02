"""Empirical (frames, torsions) marginal for one endpoint ('start' or 'end') of a
tetrapeptide's min-flux metastable state pair, for Transition Path Sampling.

The min-flux state pair itself (pi_0, pi_1) is NOT computed here -- it requires
`pyemma` (mdgen/analysis.py's TICA/k-means/MSM fit), which has no Windows wheels and
won't build against MSVC here. That step runs once, offline, via
scripts/precompute_tps_states.py under the separate `tps_eval` conda env, and caches
its (pyemma-free) output -- which frame indices belong to each state -- to
`{cache_dir}/{peptide}_states.npz`. This module only reads that cache and does the
downstream geometry conversion (mdgen.geometry, unmodified, no pyemma needed) and
ambient-vector flattening for the Product manifold defined in bridge/trainer_dbdsb.py.
"""
import os
import sys

import numpy as np
import torch

_VENDOR_MDGEN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "vendor", "mdgen")
if _VENDOR_MDGEN not in sys.path:
    sys.path.insert(0, _VENDOR_MDGEN)

from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions  # noqa: E402
from mdgen.residue_constants import restype_order  # noqa: E402

N_TORSIONS = 7  # pre-omega, phi, psi, chi1-4 (mdgen/geometry.py::atom37_to_torsions)


def root_relative_frames(frames):
    """Rigid (..., L) absolute per-residue frames -> Rigid (..., L-1) frames of
    residues 1..L-1 expressed relative to residue 0 (`frame_0^-1 . frame_i`).

    Residue 0's own absolute pose is pure gauge (arbitrary lab-frame placement of
    the whole molecule) and carries zero information about the conformation --
    dropping it makes the representation exactly invariant to any global SE(3)
    transform applied to the whole molecule (frame_i -> g.frame_i for all i
    simultaneously), the same fix bridge/data/alanine_dipeptide.py already applies
    for its single relative SO(3) frame (see that module's "v1...dominated by the
    molecule's unrestrained lab-frame tumbling" cache-versioning comment)."""
    root = frames[..., 0:1]
    rest = frames[..., 1:]
    return root.invert().compose(rest)


def flatten_state(trans, rot, torsion_angle):
    """(n_frames,3), (n_frames,3,3), (n_res,n_torsions) -> ambient vector
    (3*n_frames + 9*n_frames + n_torsions*n_res,), matching the
    Product([Euclidean(3*n_frames), SO3(n_copies=n_frames), Torus()]) manifold's
    block layout in bridge/trainer_dbdsb.py (n_frames = n_res - 1: root-relative
    frames for residues 1..L-1, root itself dropped). Batched over leading dims."""
    n_frames = trans.shape[-2]
    n_res = torsion_angle.shape[-2]
    return torch.cat([
        trans.reshape(*trans.shape[:-2], 3 * n_frames),
        rot.reshape(*rot.shape[:-3], 9 * n_frames),
        torsion_angle.reshape(*torsion_angle.shape[:-2], torsion_angle.shape[-1] * n_res),
    ], dim=-1)


def unflatten_state(x, n_frames, n_res, n_torsions=N_TORSIONS):
    """Inverse of flatten_state. x: (..., 3*n_frames + 9*n_frames + n_torsions*n_res)."""
    trans = x[..., :3 * n_frames].reshape(*x.shape[:-1], n_frames, 3)
    rot = x[..., 3 * n_frames:3 * n_frames + 9 * n_frames].reshape(*x.shape[:-1], n_frames, 3, 3)
    torsion_angle = x[..., 3 * n_frames + 9 * n_frames:].reshape(*x.shape[:-1], n_res, n_torsions)
    return trans, rot, torsion_angle


def _seqres_tensor(seqres):
    return torch.tensor([restype_order[c] for c in seqres])


def atom14_frame_to_ambient(atom14_frame, seqres_tensor):
    """atom14_frame: (L, 14, 3) numpy, one MD snapshot. Returns ambient vector (dim,)."""
    arr = atom14_frame[None].astype(np.float32)  # (1, L, 14, 3)
    frames = atom14_to_frames(torch.from_numpy(arr))  # Rigid, batch dim 1
    # aatype batch-expanded to match atom14's leading dims -- mdgen/dataset.py's
    # MDGenDataset (the code path mdgen actually trains against) does the same
    # (`aatype = ...[None].expand(num_frames, -1)`); tps_inference.py's own
    # get_sample() passes a bare (L,)-shaped seqres here instead, which does not
    # broadcast against atom14's (1, L, 14, 3) under geometry.py's batched_gather
    # and raises a shape-mismatch error -- using the dataset.py convention instead.
    aatype = seqres_tensor[None]  # (1, L)
    atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()
    torsions_sin_cos, torsion_mask = atom37_to_torsions(atom37, aatype)  # (1, L, 7, 2), (1, L, 7)

    rel_frames = root_relative_frames(frames)  # Rigid, (1, L-1)
    trans = rel_frames._trans[0]  # (L-1, 3)
    rot = rel_frames._rots._rot_mats[0]  # (L-1, 3, 3)
    torsion_angle = torch.atan2(torsions_sin_cos[0, :, :, 0], torsions_sin_cos[0, :, :, 1])  # (L, 7)
    # Structurally undefined torsions (pre-omega/phi at residue 0, which has no
    # preceding residue; any chi angle a residue type doesn't have, e.g. glycine)
    # are computed from mdgen's zero-padded "missing" atom positions and are not
    # physically meaningful -- atom37_to_torsions's own torsion_mask flags exactly
    # these. Zeroed here rather than left as whatever atan2 produced from that
    # padding, since those garbage values are not SE(3)-invariant (unlike every
    # real torsion) and would otherwise leak non-invariant noise into training.
    torsion_angle = torch.where(torsion_mask[0].bool(), torsion_angle, torch.zeros_like(torsion_angle))
    return flatten_state(trans, rot, torsion_angle)


class TetrapeptideTPSDataset(torch.utils.data.Dataset):
    """Empirical marginal (frames, torsions) for one endpoint of a tetrapeptide's
    min-flux TPS state pair. y is an unused placeholder, matching
    AlanineDipeptideFramesDataset's convention."""

    def __init__(self, peptide, endpoint, data_dir, cache_dir, n_samples, suffix="", seed=1):
        assert endpoint in ("start", "end")
        states_path = os.path.join(cache_dir, f"{peptide}_states.npz")
        if not os.path.isfile(states_path):
            raise FileNotFoundError(
                f"{states_path} not found -- run scripts/precompute_tps_states.py (under the `tps_eval` "
                f"conda env, which has pyemma) for peptide='{peptide}' first")
        states = np.load(states_path)
        idxs = states["start_idxs"] if endpoint == "start" else states["end_idxs"]
        self.state = int(states["start_state"] if endpoint == "start" else states["end_state"])

        arr = np.lib.format.open_memmap(os.path.join(data_dir, f"{peptide}{suffix}.npy"), mode="r")

        import pandas as pd
        splits_df = pd.read_csv(os.path.join(_VENDOR_MDGEN, "splits", "4AA.csv"), index_col="name")
        seqres_tensor = _seqres_tensor(splits_df.seqres[peptide])
        self.L = len(splits_df.seqres[peptide])
        self.n_frames = self.L - 1  # root-relative SE(3) frames (residue 0 dropped, see root_relative_frames)

        rng = np.random.default_rng(seed)
        drawn = rng.choice(idxs, size=n_samples, replace=True)

        points = torch.empty(n_samples, 3 * self.n_frames + 9 * self.n_frames + N_TORSIONS * self.L)
        for i, idx in enumerate(drawn):
            frame = np.copy(arr[idx]).astype(np.float32)  # (L, 14, 3)
            points[i] = atom14_frame_to_ambient(frame, seqres_tensor)
        self.points = points
        self.drawn_idxs = drawn  # raw MD frame index each sample was drawn from (metadata bookkeeping)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        return self.points[index], torch.zeros((1,))
