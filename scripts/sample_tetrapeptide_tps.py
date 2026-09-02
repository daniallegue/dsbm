"""Sample Transition Path ensembles from a trained R-IMF Riemannian bridge and export
them in exactly the file layout vendor/mdgen/scripts/analyze_peptide_tps.py expects,
so that unmodified eval script can score our generated paths as if they were mdgen's
own tps_inference.py output.

Sampling: the forward net trained by the last IPF iteration already encodes the
learned bridge from pi_0 (start state) towards pi_1 (end state) -- no separate
Doob-pinning step is needed. `DBDSB_Riemannian.record_langevin_seq(net_f, x0, None,
fb='f', num_steps=num_frames)` (the same call `test_dbdsb.py` uses generically for
any Riemannian dataset) returns the full num_frames-length interpolated trajectory
directly.

Usage (run from the repo root, in the project's main venv -- no pyemma needed here,
only mdtraj + the vendored geometry/protein/utils modules):
    python scripts/sample_tetrapeptide_tps.py \
        --checkpoint experiments/.../checkpoints/sample_net_f_002_0000020.ckpt \
        --peptide IPGD --data_dir data/4AA_data --cache_dir data/tetrapeptide_tps/IPGD \
        --out_dir results/tetrapeptide_tps/IPGD --num_batches 2 --batch_size 10 --num_frames 100
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "vendor", "mdgen"))
sys.path.insert(0, REPO_ROOT)

from mdgen.geometry import frames_torsions_to_atom14  # noqa: E402
from mdgen.rigid_utils import Rigid, Rotation  # noqa: E402
from mdgen.utils import atom14_to_pdb  # noqa: E402

from bridge.data.tetrapeptide_tps import TetrapeptideTPSDataset, unflatten_state, _seqres_tensor, N_TORSIONS  # noqa: E402
from bridge.sde.manifold import Euclidean, SO3, Torus, Product  # noqa: E402
from bridge.sde.diffusion_bridge_riemannian import DBDSB_Riemannian  # noqa: E402
from bridge.runners.config_getters import get_model  # noqa: E402

# Canonical placement for the dropped root frame (residue 0) at export time -- must
# be nonzero, see ambient_to_atom14's docstring.
_ROOT_TRANS_OFFSET = torch.tensor([20.0, 20.0, 20.0])


def build_gammas(gamma_min, gamma_max, num_steps, symmetric_gamma, gamma_space):
    if symmetric_gamma:
        n = num_steps // 2
        fn = np.linspace if gamma_space == "linspace" else np.geomspace
        gamma_half = fn(gamma_min, gamma_max, n)
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
    else:
        fn = np.linspace if gamma_space == "linspace" else np.geomspace
        gammas = fn(gamma_min, gamma_max, num_steps)
    return torch.tensor(gammas).float()


def build_langevin(cfg, L, n_torsions):
    n_frames = L - 1  # root-relative SE(3) frames; residue 0's absolute pose dropped
                       # (see bridge/data/tetrapeptide_tps.py::root_relative_frames)
    gammas = build_gammas(cfg["gamma_min"], cfg["gamma_max"], cfg["num_steps"],
                           cfg["symmetric_gamma"], cfg["gamma_space"])
    T = torch.sum(gammas)
    sigma = torch.sqrt(T).item()
    timesteps = gammas / T

    manifold = Product([
        (Euclidean(3 * n_frames), 3 * n_frames),
        (SO3(n_copies=n_frames), 9 * n_frames),
        (Torus(), n_torsions * L),
    ])
    dim = 3 * n_frames + 9 * n_frames + n_torsions * L
    langevin = DBDSB_Riemannian(
        sigma, cfg["num_steps"], timesteps, (dim,), (1,), cfg["first_coupling"], cfg["mean_match"],
        loss_type=cfg["loss_type"], gamma_min=cfg["gamma_min"], gamma_max=cfg["gamma_max"],
        symmetric_gamma=cfg["symmetric_gamma"], gamma_space=cfg["gamma_space"], manifold=manifold)
    return langevin


def ambient_to_atom14(x, L, n_torsions, seqres_tensor):
    """x: (N, dim) ambient vectors (root-relative frames for residues 1..L-1, plus
    all L residues' torsions) -> atom14 positions (N, L, 14, 3) numpy (Angstrom).

    Reconstructs absolute frames by placing residue 0 at a fixed canonical pose (the
    dropped global pose was pure gauge to begin with, see root_relative_frames):
    frame_0 = (identity rotation, _ROOT_TRANS_OFFSET translation), frame_i =
    frame_0 . rel_i for i=1..L-1. Downstream (mdtraj superpose, torsion-based
    featurization) is invariant to this choice, so it has no effect on anything but
    the arbitrary orientation/position the output PDB is written in -- EXCEPT that
    the offset must be nonzero: mdgen/utils.py::create_full_prot treats any atom
    with near-zero coordinates as "missing" (`atom37_mask = sum(abs(atom37)) >
    1e-7`), so placing residue 0's CA exactly at the origin silently drops its
    backbone atoms from the written PDB and corrupts downstream featurization."""
    n_frames = L - 1
    rel_trans, rel_rot, torsion_angle = unflatten_state(x, n_frames, L, n_torsions=n_torsions)
    N = x.shape[0]
    root_offset = _ROOT_TRANS_OFFSET.to(dtype=rel_trans.dtype, device=rel_trans.device)
    root_trans = root_offset.expand(N, 1, 3)
    root_rot = torch.eye(3, dtype=rel_rot.dtype, device=rel_rot.device).expand(N, 1, 3, 3)
    # frame_0 has identity rotation, so composing it onto residues 1..L-1 only adds
    # the translation offset (rotation part of rel_rot is unchanged by an identity
    # left-rotation); see the docstring above for why this offset can't be zero.
    trans = torch.cat([root_trans, rel_trans + root_offset], dim=1)  # (N, L, 3)
    rot = torch.cat([root_rot, rel_rot], dim=1)  # (N, L, 3, 3)

    torsion_sin_cos = torch.stack([torch.sin(torsion_angle), torch.cos(torsion_angle)], dim=-1)
    frames = Rigid(Rotation(rot_mats=rot), trans)
    aatype = seqres_tensor[None].expand(N, -1)
    atom14 = frames_torsions_to_atom14(frames, torsion_sin_cos, aatype)
    return atom14.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a sample_net_f_*.ckpt")
    parser.add_argument("--peptide", required=True)
    parser.add_argument("--data_dir", default="data/4AA_data")
    parser.add_argument("--cache_dir", required=True, help="dir with {peptide}_states.npz and _metadata.pkl "
                                                             "(from scripts/precompute_tps_states.py)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_residues", type=int, default=4)
    parser.add_argument("--n_torsions", type=int, default=N_TORSIONS)
    parser.add_argument("--num_steps", type=int, default=10, help="R-IMF training num_steps (bridge discretization)")
    parser.add_argument("--gamma_min", type=float, default=0.001)
    parser.add_argument("--gamma_max", type=float, default=0.05)
    parser.add_argument("--symmetric_gamma", action="store_true", default=True)
    parser.add_argument("--gamma_space", default="linspace")
    parser.add_argument("--first_coupling", default="ind")
    parser.add_argument("--mean_match", action="store_true", default=True)
    parser.add_argument("--loss_type", default="log_bridge")
    parser.add_argument("--encoder_layers", type=int, nargs="+", default=[128])
    parser.add_argument("--decoder_layers", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--temb_dim", type=int, default=64)
    parser.add_argument("--temb_max_period", type=int, default=10000)
    parser.add_argument("--num_frames", type=int, default=100, help="output trajectory length, "
                                                                      "matches mdgen's own TPS num_frames")
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    import pandas as pd
    splits_df = pd.read_csv(os.path.join(REPO_ROOT, "vendor", "mdgen", "splits", "4AA.csv"), index_col="name")
    seqres_tensor = _seqres_tensor(splits_df.seqres[args.peptide])
    L = len(splits_df.seqres[args.peptide])
    assert L == args.n_residues

    dim = 3 * (L - 1) + 9 * (L - 1) + args.n_torsions * L

    class _ModelCfg:
        pass
    model_cfg = _ModelCfg()
    model_cfg.model = _ModelCfg()
    model_cfg.model.encoder_layers = args.encoder_layers
    model_cfg.model.decoder_layers = args.decoder_layers
    model_cfg.model.temb_dim = args.temb_dim
    model_cfg.model.temb_max_period = args.temb_max_period
    model_cfg.data = _ModelCfg()
    model_cfg.data.dim = dim
    model_cfg.Model = "Basic"

    net_f = get_model(model_cfg)
    net_f.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    net_f = net_f.to(args.device).eval()

    langevin_cfg = dict(num_steps=args.num_steps, gamma_min=args.gamma_min, gamma_max=args.gamma_max,
                         symmetric_gamma=args.symmetric_gamma, gamma_space=args.gamma_space,
                         first_coupling=args.first_coupling, mean_match=args.mean_match, loss_type=args.loss_type)
    langevin = build_langevin(langevin_cfg, L, args.n_torsions)
    langevin.device = args.device

    init_ds = TetrapeptideTPSDataset(peptide=args.peptide, endpoint="start", data_dir=args.data_dir,
                                      cache_dir=args.cache_dir, n_samples=args.num_batches * args.batch_size,
                                      seed=args.seed + 1)
    final_ds = TetrapeptideTPSDataset(peptide=args.peptide, endpoint="end", data_dir=args.data_dir,
                                       cache_dir=args.cache_dir, n_samples=args.num_batches * args.batch_size,
                                       seed=args.seed + 2)

    def net_fn(x, y, t):
        return net_f(x, y, t)

    metadata = []
    with torch.no_grad():
        for b in range(args.num_batches):
            sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
            x0 = init_ds.points[sl].to(args.device)
            x_tot, _, _, _ = langevin.record_langevin_seq(net_fn, x0, None, fb="f", num_steps=args.num_frames)
            # x_tot: (batch_size, num_frames, dim)
            for j in range(x0.shape[0]):
                idx = b * args.batch_size + j
                atom14 = ambient_to_atom14(x_tot[j], L, args.n_torsions, seqres_tensor)  # (num_frames, L, 14, 3)
                path = os.path.join(args.out_dir, f"{args.peptide}_{idx}.pdb")
                atom14_to_pdb(atom14, seqres_tensor.cpu().numpy(), path)

                import mdtraj
                traj = mdtraj.load(path)
                traj.superpose(traj)
                traj.save(os.path.join(args.out_dir, f"{args.peptide}_{idx}.xtc"))
                traj[0].save(path)

                metadata.append({
                    "name": args.peptide,
                    "start_idx": int(init_ds.drawn_idxs[idx]),
                    "end_idx": int(final_ds.drawn_idxs[idx]),
                    "start_state": init_ds.state,
                    "end_state": final_ds.state,
                    "path": path,
                })
            print(f"batch {b + 1}/{args.num_batches} done")

    with open(os.path.join(args.out_dir, f"{args.peptide}_metadata.json"), "w") as f:
        json.dump(metadata, f)

    metadata_pkl_src = os.path.join(args.cache_dir, f"{args.peptide}_metadata.pkl")
    metadata_pkl_dst = os.path.join(args.out_dir, f"{args.peptide}_metadata.pkl")
    if os.path.isfile(metadata_pkl_src) and not os.path.isfile(metadata_pkl_dst):
        shutil.copy(metadata_pkl_src, metadata_pkl_dst)

    print(f"wrote {len(metadata)} trajectories to {args.out_dir}")


if __name__ == "__main__":
    main()
