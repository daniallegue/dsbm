import argparse
import glob
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bridge.models import ScoreNetwork
from bridge.sde import DBDSB_Riemannian, DBDSB_VE, SO3
from bridge.data.alanine_dipeptide import (AlanineDipeptideFramesDataset, BASIN_CENTERS_DEG,
                                            calibrate_bond_angles, frames_to_phi_psi)

ROOT = "experiments/alanine_dipeptide_AlanineDipeptideFrames/loss_type=log_bridge,method=dbdsb_riemannian,model=Basic/42/checkpoints"
DATA_CACHE_DIR = "data/alanine_dipeptide"


def model_kwargs(x_dim):
    return dict(encoder_layers=[128], temb_dim=64, decoder_layers=[512, 512, 512], x_dim=x_dim, temb_max_period=10000)


def build_langevin_riemannian(x_dim, manifold, num_steps, gamma_min, gamma_max, loss_type="log_bridge", first_coupling="ind"):
    """Matches conf/method/dbdsb_riemannian.yaml: symmetric (triangular) gamma schedule."""
    n = num_steps // 2
    gh = np.linspace(gamma_min, gamma_max, n)
    gammas = np.concatenate([gh, np.flip(gh)])
    gammas = torch.tensor(gammas).float()
    T = gammas.sum()
    sigma = torch.sqrt(T).item()
    timesteps = gammas / T
    return DBDSB_Riemannian(sigma, num_steps, timesteps, (x_dim,), (1,), first_coupling=first_coupling, mean_match=False,
                             loss_type=loss_type, gamma_min=gamma_min, gamma_max=gamma_max,
                             symmetric_gamma=True, gamma_space="linspace", manifold=manifold)


def build_langevin_ve(x_dim, num_steps, gamma_min, gamma_max):
    """Matches conf/method/dbdsb.yaml: monotonic (non-symmetric) gamma schedule, flat R^d."""
    gammas = torch.tensor(np.linspace(gamma_min, gamma_max, num_steps)).float()
    T = gammas.sum()
    sigma = torch.sqrt(T).item()
    timesteps = gammas / T
    return DBDSB_VE(sigma, num_steps, timesteps, (x_dim,), (1,), first_coupling="ref", mean_match=False)


def basin_occupancy(phi_deg, psi_deg, center_deg, radius_rad=0.75):
    """Fraction of (phi, psi) points within radius_rad (wrapped) of center_deg -- the
    same TPS-DPS state-ball convention used to define the basins in the MD data --
    plus the mean wrapped distance to the center, in degrees."""
    phi, psi = np.deg2rad(phi_deg), np.deg2rad(psi_deg)
    c_phi, c_psi = np.deg2rad(center_deg[0]), np.deg2rad(center_deg[1])
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    dist = np.sqrt(wrap(phi - c_phi) ** 2 + wrap(psi - c_psi) ** 2)
    return (dist < radius_rad).mean(), np.rad2deg(dist.mean())


def manifold_adherence(x_flat):
    """How far generated (n, 9) vectors, reshaped as 3x3 matrices, deviate from SO(3):
    Frobenius norm of R^T R - I (orthogonality) and |det(R) - 1| (orientation/scale).
    Should be ~0 for a manifold-constrained sampler; a flat Euclidean bridge has no
    mechanism to keep samples on SO(3), so this quantifies exactly what that costs."""
    r = x_flat.numpy().reshape(-1, 3, 3)
    orth_err = np.abs(np.matmul(r.transpose(0, 2, 1), r) - np.eye(3)).reshape(r.shape[0], -1).sum(axis=-1)
    det_err = np.abs(np.linalg.det(r) - 1.0)
    return orth_err.mean(), det_err.mean()


def find_latest_ckpt(ckpt_dir, fb):
    """Highest (ipf_iter, step) sample_net_{fb}_*.ckpt in ckpt_dir -- avoids silently
    picking up an intermediate checkpoint whose (n, i) happens to also exist (e.g. an
    earlier run's n=010 file is a real, valid, but non-final checkpoint of a longer run
    that later reached n=020)."""
    pattern = re.compile(rf"sample_net_{fb}_(\d+)_(\d+)\.ckpt$")
    candidates = []
    for path in glob.glob(os.path.join(ckpt_dir, f"sample_net_{fb}_*.ckpt")):
        m = pattern.search(os.path.basename(path))
        if m:
            candidates.append(((int(m.group(1)), int(m.group(2))), path))
    if not candidates:
        raise FileNotFoundError(f"no sample_net_{fb}_*.ckpt found in {ckpt_dir}")
    candidates.sort()
    return candidates[-1]  # ((n, i), path)


def load_net(ckpt_path, x_dim):
    net = ScoreNetwork(**model_kwargs(x_dim))
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    net.eval()
    return net


def simulate_trajectory(langevin, net, x0, fb):
    with torch.no_grad():
        x_traj, _, _, _ = langevin.record_langevin_seq(net, x0, None, fb, sample=False)
    return x_traj[:, -1]  # (N, 9) -- final generated endpoint


def load_basin(basin, n_samples, seed):
    ds = AlanineDipeptideFramesDataset(basin=basin, n_samples=n_samples, cache_dir=DATA_CACHE_DIR,
                                        temperature_k=300.0, friction_per_ps=1.0, save_every=10,
                                        ball_radius=0.75, seed=seed)
    return torch.stack([ds[i][0] for i in range(len(ds))])


def to_phi_psi_deg(x_flat, theta_n, theta_c):
    frames = x_flat.numpy().reshape(-1, 3, 3)
    phi, psi = frames_to_phi_psi(frames, theta_n, theta_c)
    return np.rad2deg(phi), np.rad2deg(psi)


def panel(ax, ref_phi, ref_psi, ref_color, ref_label, gen_phi, gen_psi, gen_label, basin_center, title):
    ax.scatter(ref_phi, ref_psi, c=ref_color, s=6, alpha=0.25, label=ref_label)
    ax.scatter(gen_phi, gen_psi, c="firebrick", s=10, alpha=0.7, label=gen_label)
    ax.scatter([basin_center[0]], [basin_center[1]], marker="*", c="black", s=180, zorder=5, label="target basin center")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel(r"$\phi$ (deg)"); ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=7, markerscale=1.5)
    ax.grid(alpha=0.2)


def main(ckpt_root, kind, loss_type, num_steps, gamma_min, gamma_max, n_samples, out_png, title):
    torch.manual_seed(0)
    x_dim = 9
    if kind == "riemannian":
        first_coupling = "ref" if loss_type == "score_divergence" else "ind"
        langevin = build_langevin_riemannian(x_dim, SO3(n_copies=1), num_steps, gamma_min, gamma_max,
                                              loss_type=loss_type, first_coupling=first_coupling)
    else:
        langevin = build_langevin_ve(x_dim, num_steps, gamma_min, gamma_max)

    print("calibrating local bond angles...")
    theta_n, theta_c = calibrate_bond_angles()

    print("loading basin reference data (cached, from training)...")
    x_c7eq = load_basin("c7eq", n_samples, seed=1)
    x_c7ax = load_basin("c7ax", n_samples, seed=2)
    ref_phi_eq, ref_psi_eq = to_phi_psi_deg(x_c7eq, theta_n, theta_c)
    ref_phi_ax, ref_psi_ax = to_phi_psi_deg(x_c7ax, theta_n, theta_c)

    (f_n, f_i), f_path = find_latest_ckpt(ckpt_root, "f")
    (b_n, b_i), b_path = find_latest_ckpt(ckpt_root, "b")
    print(f"loading checkpoints: f=(ipf={f_n}, step={f_i}), b=(ipf={b_n}, step={b_i})")
    net_f = load_net(f_path, x_dim)
    net_b = load_net(b_path, x_dim)

    gen_from_eq = simulate_trajectory(langevin, net_f, x_c7eq, "f")   # should land near c7ax
    gen_from_ax = simulate_trajectory(langevin, net_b, x_c7ax, "b")   # should land near c7eq
    gen_phi_f, gen_psi_f = to_phi_psi_deg(gen_from_eq, theta_n, theta_c)
    gen_phi_b, gen_psi_b = to_phi_psi_deg(gen_from_ax, theta_n, theta_c)

    occ_f, dist_f = basin_occupancy(gen_phi_f, gen_psi_f, BASIN_CENTERS_DEG["c7ax"])
    occ_b, dist_b = basin_occupancy(gen_phi_b, gen_psi_b, BASIN_CENTERS_DEG["c7eq"])
    orth_f, det_f = manifold_adherence(gen_from_eq)
    orth_b, det_b = manifold_adherence(gen_from_ax)
    print(f"[{title}] forward  (C7eq->C7ax): basin occupancy={occ_f:.1%}  mean dist to center={dist_f:.1f} deg  "
          f"|R^T R - I|={orth_f:.4f}  |det(R)-1|={det_f:.4f}")
    print(f"[{title}] backward (C7ax->C7eq): basin occupancy={occ_b:.1%}  mean dist to center={dist_b:.1f} deg  "
          f"|R^T R - I|={orth_b:.4f}  |det(R)-1|={det_b:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    panel(axes[0], ref_phi_eq, ref_psi_eq, "silver", "real C7eq (source)",
          gen_phi_f, gen_psi_f, "generated (f: C7eq→C7ax)",
          BASIN_CENTERS_DEG["c7ax"], "Forward: C7eq → C7ax")
    panel(axes[1], ref_phi_ax, ref_psi_ax, "lightsteelblue", "real C7ax (source)",
          gen_phi_b, gen_psi_b, "generated (b: C7ax→C7eq)",
          BASIN_CENTERS_DEG["c7eq"], "Backward: C7ax → C7eq")

    fig.suptitle(f"{title}: projected to Ramachandran space", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("saved", out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ramachandran-plot check of a trained alanine dipeptide bridge.")
    parser.add_argument("--checkpoint-dir", default=ROOT)
    parser.add_argument("--kind", choices=["riemannian", "ve"], default="riemannian",
                         help="riemannian = conf/method/dbdsb_riemannian.yaml (SO(3)); ve = conf/method/dbdsb.yaml (flat Euclidean)")
    parser.add_argument("--loss-type", choices=["log_bridge", "score_divergence", "bridge_matching"], default="log_bridge",
                         help="riemannian only")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--gamma-min", type=float, default=0.001)
    parser.add_argument("--gamma-max", type=float, default=None,
                         help="default 0.05 for --kind riemannian (symmetric schedule), 0.2 for --kind ve (monotonic) -- matching each method's yaml")
    parser.add_argument("--n-samples", type=int, default=2500,
                         help="must match the training dataset's n_samples to hit its cache (see conf/dataset/alanine_dipeptide.yaml)")
    parser.add_argument("--out", default="alanine_bridge_ramachandran.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        raise FileNotFoundError(f"checkpoint dir not found: {args.checkpoint_dir}")

    gamma_max = args.gamma_max if args.gamma_max is not None else (0.05 if args.kind == "riemannian" else 0.2)
    title = args.title if args.title is not None else f"Alanine dipeptide, {args.kind}" + (f"/{args.loss_type}" if args.kind == "riemannian" else "")

    main(args.checkpoint_dir, args.kind, args.loss_type, args.num_steps, args.gamma_min, gamma_max,
         args.n_samples, args.out, title)
