import torch
import numpy as np
from scipy.special import sph_harm_y


def sample_uniform_sphere(n, d=3, generator=None):
    x = torch.randn(n, d, generator=generator)
    return x / x.norm(dim=-1, keepdim=True)


def real_spherical_harmonic(l, m, x):
    """Re(Y_l^m(x)) for unit vectors x: (N, 3) -> (N,)."""
    x_np = x.numpy()
    theta = np.arccos(np.clip(x_np[:, 2], -1., 1.))       # polar angle from z-axis
    phi = np.arctan2(x_np[:, 1], x_np[:, 0]) % (2 * np.pi)  # azimuthal angle
    y = sph_harm_y(l, m, theta, phi)
    return torch.from_numpy(np.real(y)).float()


@torch.no_grad()
def sample_spherical_harmonic(l, m, n_samples, batch_mult=4, max_tries=500, generator=None):
    """Rejection-sample points on S^2 with density proportional to |Re(Y_l^m)|,
    matching the RDSB paper's "samples on the sphere taken with probability
    proportional to the real component of [a] spherical harmonic function"."""
    probe = sample_uniform_sphere(200_000, generator=generator)
    M = real_spherical_harmonic(l, m, probe).abs().max().item() * 1.05
    assert M > 0, f"Re(Y_{l}^{m}) is identically zero (need |m| <= l)"

    collected = []
    n_collected = 0
    tries = 0
    while n_collected < n_samples and tries < max_tries:
        cand = sample_uniform_sphere(n_samples * batch_mult, generator=generator)
        density = real_spherical_harmonic(l, m, cand).abs()
        u = torch.rand(cand.shape[0], generator=generator) * M
        accepted = cand[u < density]
        collected.append(accepted)
        n_collected += accepted.shape[0]
        tries += 1

    result = torch.cat(collected, dim=0)
    assert result.shape[0] >= n_samples, "rejection sampling did not converge; increase max_tries/batch_mult"
    return result[:n_samples]


class SphericalHarmonicDataset(torch.utils.data.Dataset):
    """Points on S^2 sampled with density proportional to |Re(Y_l^m)|.
    y is an unused placeholder (this dataset is unconditional)."""

    def __init__(self, l, m, n_samples, seed=None):
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.points = sample_spherical_harmonic(l, m, n_samples, generator=generator)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        return self.points[index], torch.zeros((1,))
