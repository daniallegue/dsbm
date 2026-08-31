import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bridge.models import ScoreNetwork
from bridge.sde import DBDSB_Riemannian, Sphere, Torus
from bridge.data.spherical_harmonics import SphericalHarmonicDataset
from bridge.data.torus import TorusMixtureDataset, sample_torus_mixture

TORUS_R, TORUS_r = 1.5, 0.6  # major/minor radius for the 3D donut embedding

SPHERE_ROOT = "experiments/spherical_harmonics_SphericalHarmonics"
TORUS_ROOT = "experiments/torus_gaussians_TorusGaussians"

DEFAULT_CHECKPOINTS = {
    "sphere": [
        ("score_divergence", f"{SPHERE_ROOT}/method=dbdsb_riemannian,model=Basic/42/checkpoints/sample_net_f_010_0002000.ckpt"),
        ("log_bridge (ind)", f"{SPHERE_ROOT}/loss_type=log_bridge,method=dbdsb_riemannian,model=Basic/42/checkpoints/sample_net_f_010_0002000.ckpt"),
        ("log_bridge (ref)", f"{SPHERE_ROOT}/first_coupling=ref,method=dbdsb_riemannian,model=Basic/42/checkpoints/sample_net_f_010_0002000.ckpt"),
    ],
    "torus": [
        ("log_bridge (ind)", f"{TORUS_ROOT}/method=dbdsb_riemannian,model=Basic/42/checkpoints/sample_net_f_010_0002000.ckpt"),
        ("log_bridge (ref)", f"{TORUS_ROOT}/first_coupling=ref,method=dbdsb_riemannian,model=Basic/42/checkpoints/sample_net_f_010_0002000.ckpt"),
        ("score_divergence (ref)", f"{TORUS_ROOT}/first_coupling=ref,loss_type=score_divergence,method=dbdsb_riemannian,model=Basic/42/checkpoints/sample_net_f_010_0002000.ckpt"),
    ],
}


def model_kwargs(x_dim):
    return dict(encoder_layers=[128], temb_dim=64, decoder_layers=[512, 512, 512], x_dim=x_dim, temb_max_period=10000)


def build_langevin(x_dim, manifold, num_steps, gamma_min, gamma_max):
    n = num_steps // 2
    gh = np.linspace(gamma_min, gamma_max, n)
    gammas = np.concatenate([gh, np.flip(gh)])
    gammas = torch.tensor(gammas).float()
    T = gammas.sum()
    sigma = torch.sqrt(T).item()
    timesteps = gammas / T
    return DBDSB_Riemannian(sigma, num_steps, timesteps, (x_dim,), (1,), first_coupling="ind", mean_match=False,
                             loss_type="log_bridge", gamma_min=gamma_min, gamma_max=gamma_max,
                             symmetric_gamma=True, gamma_space="linspace", manifold=manifold)


def load_net(ckpt_path, x_dim):
    net = ScoreNetwork(**model_kwargs(x_dim))
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    net.eval()
    return net


def simulate_trajectory(langevin, net, x0):
    with torch.no_grad():
        x_traj, _, _, _ = langevin.record_langevin_seq(net, x0, None, "f", sample=False)
    return torch.cat([x0.unsqueeze(1), x_traj], dim=1).numpy()  # (N, num_steps+1, d)


def torus_embed(points):
    theta, phi = points[..., 0], points[..., 1]
    x = (TORUS_R + TORUS_r * np.cos(phi)) * np.cos(theta)
    y = (TORUS_R + TORUS_r * np.cos(phi)) * np.sin(theta)
    z = TORUS_r * np.sin(phi)
    return np.stack([x, y, z], axis=-1)


def draw_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightgrey", alpha=0.10, linewidth=0, shade=False)
    u_grid = np.linspace(0, 2 * np.pi, 13)
    v_grid = np.linspace(0, np.pi, 7)
    xs_g = np.outer(np.cos(u_grid), np.sin(v_grid))
    ys_g = np.outer(np.sin(u_grid), np.sin(v_grid))
    zs_g = np.outer(np.ones_like(u_grid), np.cos(v_grid))
    ax.plot_wireframe(xs_g, ys_g, zs_g, color="white", linewidth=0.5, alpha=0.6)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75); ax.set_zlim(-0.75, 0.75)


def draw_torus(ax):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    xs = (TORUS_R + TORUS_r * np.cos(V)) * np.cos(U)
    ys = (TORUS_R + TORUS_r * np.cos(V)) * np.sin(U)
    zs = TORUS_r * np.sin(V)
    ax.plot_surface(xs, ys, zs, color="lightgrey", alpha=0.10, linewidth=0, shade=False)
    u_grid = np.linspace(0, 2 * np.pi, 13)
    v_grid = np.linspace(0, 2 * np.pi, 9)
    Ug, Vg = np.meshgrid(u_grid, v_grid)
    xs_g = (TORUS_R + TORUS_r * np.cos(Vg)) * np.cos(Ug)
    ys_g = (TORUS_R + TORUS_r * np.cos(Vg)) * np.sin(Ug)
    zs_g = TORUS_r * np.sin(Vg)
    ax.plot_wireframe(xs_g, ys_g, zs_g, color="white", linewidth=0.5, alpha=0.6)
    lim = TORUS_R + TORUS_r
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)


def panel(ax, traj_3d, target_ref_3d, title, elev, azim, draw_surface):
    draw_surface(ax)
    ax.scatter(target_ref_3d[:, 0], target_ref_3d[:, 1], target_ref_3d[:, 2], c="silver", s=2, alpha=0.25, depthshade=False)

    n_lines = min(25, traj_3d.shape[0])
    idx = np.random.default_rng(0).choice(traj_3d.shape[0], n_lines, replace=False)
    for i in idx:
        ax.plot(traj_3d[i, :, 0], traj_3d[i, :, 1], traj_3d[i, :, 2], color="firebrick", alpha=0.35, linewidth=0.8)

    final = traj_3d[:, -1]
    ax.scatter(final[:, 0], final[:, 1], final[:, 2], c="firebrick", s=9, alpha=0.9, depthshade=False)

    ax.set_title(title, fontsize=11, pad=-4)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)


def main_sphere(checkpoints, l_init, m_init, l_final, m_final, num_steps, gamma_min, gamma_max, n_samples, elev, azim, out_png):
    torch.manual_seed(0)
    langevin = build_langevin(3, Sphere(), num_steps, gamma_min, gamma_max)
    init_ds = SphericalHarmonicDataset(l_init, m_init, n_samples, seed=1)
    final_ds = SphericalHarmonicDataset(l_final, m_final, n_samples, seed=2)
    x0 = torch.stack([init_ds[i][0] for i in range(n_samples)])
    target_ref = torch.stack([final_ds[i][0] for i in range(n_samples)]).numpy()

    fig = plt.figure(figsize=(5.5 * len(checkpoints), 5.5))
    for i, (name, ckpt) in enumerate(checkpoints):
        net = load_net(ckpt, 3)
        traj = simulate_trajectory(langevin, net, x0)
        ax = fig.add_subplot(1, len(checkpoints), i + 1, projection="3d")
        panel(ax, traj, target_ref, name, elev, azim, draw_sphere)

    fig.suptitle(f"Transition: source (l={l_init},m={m_init}) -> target (l={l_final},m={m_final})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("saved", out_png)


def main_torus(checkpoints, n_modes_init, n_modes_final, kappa, ring_radius, num_steps, gamma_min, gamma_max, n_samples, elev, azim, out_png):
    torch.manual_seed(0)
    langevin = build_langevin(2, Torus(), num_steps, gamma_min, gamma_max)
    phase_offset = np.pi / n_modes_final
    init_ds = TorusMixtureDataset(n_modes_init, kappa, n_samples, ring_radius, seed=1)
    x0 = torch.stack([init_ds[i][0] for i in range(n_samples)])
    target_ref = sample_torus_mixture(n_samples, n_modes_final, kappa, ring_radius, phase_offset, seed=2).numpy()
    target_ref_3d = torus_embed(target_ref)

    fig = plt.figure(figsize=(5.5 * len(checkpoints), 5.5))
    for i, (name, ckpt) in enumerate(checkpoints):
        net = load_net(ckpt, 2)
        traj = simulate_trajectory(langevin, net, x0)
        traj_3d = torus_embed(traj)
        ax = fig.add_subplot(1, len(checkpoints), i + 1, projection="3d")
        panel(ax, traj_3d, target_ref_3d, name, elev, azim, draw_torus)

    fig.suptitle(f"Transition: source ({n_modes_init} modes) -> target ({n_modes_final} modes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("saved", out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D manifold transitions for one or more trained checkpoints.")
    parser.add_argument("--manifold", choices=["sphere", "torus"], default="sphere")
    parser.add_argument("--checkpoint", action="append", metavar="NAME=PATH",
                         help="repeatable; defaults to the runs under experiments/ for the chosen --manifold")
    parser.add_argument("--l-init", type=int, default=4)
    parser.add_argument("--m-init", type=int, default=2)
    parser.add_argument("--l-final", type=int, default=6)
    parser.add_argument("--m-final", type=int, default=2)
    parser.add_argument("--n-modes-init", type=int, default=4)
    parser.add_argument("--n-modes-final", type=int, default=8)
    parser.add_argument("--kappa", type=float, default=8.0)
    parser.add_argument("--ring-radius", type=float, default=2.0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--gamma-min", type=float, default=0.001)
    parser.add_argument("--gamma-max", type=float, default=0.05)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--elev", type=float, default=None)
    parser.add_argument("--azim", type=float, default=None)
    parser.add_argument("--out", default="transition.png")
    args = parser.parse_args()

    checkpoints = [tuple(c.split("=", 1)) for c in args.checkpoint] if args.checkpoint else DEFAULT_CHECKPOINTS[args.manifold]
    for name, path in checkpoints:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint for '{name}' not found: {path}")

    if args.manifold == "sphere":
        elev = 65 if args.elev is None else args.elev
        azim = 20 if args.azim is None else args.azim
        main_sphere(checkpoints, args.l_init, args.m_init, args.l_final, args.m_final,
                    args.num_steps, args.gamma_min, args.gamma_max, args.n_samples, elev, azim, args.out)
    else:
        elev = 55 if args.elev is None else args.elev
        azim = 25 if args.azim is None else args.azim
        main_torus(checkpoints, args.n_modes_init, args.n_modes_final, args.kappa, args.ring_radius,
                   args.num_steps, args.gamma_min, args.gamma_max, args.n_samples, elev, azim, args.out)
