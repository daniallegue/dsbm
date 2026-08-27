import torch


class Manifold:
    """Embedded Riemannian manifold, as used by the geodesic random walk of Algorithm 1
    in De Bortoli et al., "Riemannian Score-Based Generative Modelling" (arXiv:2207.03024)."""

    def proj_tangent(self, x, v):
        """Project ambient vector(s) v onto the tangent space at x. x, v: (..., d)."""
        raise NotImplementedError

    def exp(self, x, v):
        """Riemannian exponential map: move from x along tangent vector v. x, v: (..., d)."""
        raise NotImplementedError

    def log(self, x, y):
        """Riemannian logarithm map: tangent vector at x pointing towards y,
        with norm equal to the geodesic distance d(x, y). x, y: (..., d)."""
        raise NotImplementedError


class Sphere(Manifold):
    """Unit sphere S^{d-1} embedded in R^d (last dim of x), radius 1."""

    def __init__(self, eps=1e-6):
        self.eps = eps

    def proj_tangent(self, x, v):
        return v - (x * v).sum(dim=-1, keepdim=True) * x

    def exp(self, x, v):
        norm_v = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.cos(norm_v) * x + torch.sin(norm_v) * (v / norm_v)

    def log(self, x, y):
        v = self.proj_tangent(x, y)
        norm_v = v.norm(dim=-1, keepdim=True)
        cos_theta = (x * y).sum(dim=-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.acos(cos_theta)
        # near x == y, norm_v underflows to ~0 while acos's clamp still yields a small
        return torch.where(norm_v < self.eps, torch.zeros_like(v), theta * v / norm_v.clamp_min(self.eps))


@torch.no_grad()
def geodesic_random_walk(manifold, x0, f_fn, g_fn, timesteps):
    """Algorithm 1: simulate a diffusion on `manifold` starting at x0.

    x0: (N, ..., d) initial states on the manifold
    f_fn: callable (t, x) -> tangent drift f(t, X_k) at x, same shape as x
    g_fn: callable (t) -> scalar (or broadcastable) diffusion coefficient g(t)
    timesteps: (num_steps,) step sizes gamma_k

    Returns the trajectory, shape (N, num_steps, ..., d).
    """
    x = x0
    t = torch.zeros(x.shape[0], device=x.device)
    num_steps = timesteps.shape[0]
    traj = torch.empty(x.shape[0], num_steps, *x.shape[1:], device=x.device)

    for k in range(num_steps):
        gamma = timesteps[k]
        z_bar = torch.randn_like(x)
        z = manifold.proj_tangent(x, z_bar)
        drift = f_fn(t, x)
        w = gamma * drift + torch.sqrt(gamma) * g_fn(t) * z
        x = manifold.exp(x, w)
        traj[:, k] = x
        t = t + gamma

    return traj
