import torch
import numpy as np


def sample_torus_mixture(n_samples, n_modes=8, kappa=8.0, ring_radius=2.0, phase_offset=0.0, seed=None):
    rng = np.random.default_rng(seed)
    angles = phase_offset + np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers_theta = ring_radius * np.cos(angles)
    centers_phi = ring_radius * np.sin(angles)

    mode_idx = rng.integers(0, n_modes, size=n_samples)
    theta = rng.vonmises(centers_theta[mode_idx], kappa)
    phi = rng.vonmises(centers_phi[mode_idx], kappa)
    return torch.from_numpy(np.stack([theta, phi], axis=-1)).float()


class TorusMixtureDataset(torch.utils.data.Dataset):
    """Points on the flat torus, angles in [-pi, pi), sampled from a von Mises
    mixture. y is an unused placeholder (this dataset is unconditional)."""

    def __init__(self, n_modes, kappa, n_samples, ring_radius=2.0, phase_offset=0.0, seed=None):
        self.points = sample_torus_mixture(n_samples, n_modes=n_modes, kappa=kappa, ring_radius=ring_radius,
                                            phase_offset=phase_offset, seed=seed)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        return self.points[index], torch.zeros((1,))
