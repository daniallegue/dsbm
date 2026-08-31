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


class Torus(Manifold):
    """Flat torus (R/2piZ)^d, points and tangent vectors both represented as angle
    vectors in [-pi, pi) (last dim of x)."""

    @staticmethod
    def _wrap(theta):
        return (theta + torch.pi) % (2 * torch.pi) - torch.pi

    def proj_tangent(self, x, v):
        return v

    def exp(self, x, v):
        return self._wrap(x + v)

    def log(self, x, y):
        return self._wrap(y - x)


class SO3(Manifold):
    """
    Product SO(3)^n_copies, each factor embedded in R^{3x3} with the ambient
    Frobenius inner product
    """

    def __init__(self, n_copies=1, eps=1e-6):
        self.n_copies = n_copies
        self.eps = eps

    def _unflatten(self, x):
        return x.reshape(*x.shape[:-1], self.n_copies, 3, 3)

    def _flatten(self, r):
        return r.reshape(*r.shape[:-3], self.n_copies * 9)

    @staticmethod
    def _skew(a):
        return 0.5 * (a - a.transpose(-1, -2))

    @staticmethod
    def _vee(a):
        return torch.stack([a[..., 2, 1], a[..., 0, 2], a[..., 1, 0]], dim=-1)

    def proj_tangent(self, x, v):
        x = self._unflatten(x)
        v = self._unflatten(v)
        a = self._skew(x.transpose(-1, -2) @ v)
        return self._flatten(x @ a)

    def exp(self, x, v):
        x = self._unflatten(x)
        v = self._unflatten(v)
        a = self._skew(x.transpose(-1, -2) @ v)  # skew generator, so x @ expm(a) is the geodesic
        theta = self._vee(a).norm(dim=-1, keepdim=True).clamp_min(self.eps)[..., None]  # (..., n, 1, 1)
        sinc = torch.sin(theta) / theta
        cosc = (1 - torch.cos(theta)) / theta**2
        eye = torch.eye(3, device=x.device, dtype=x.dtype).expand_as(a)
        r_delta = eye + sinc * a + cosc * (a @ a)  # Rodrigues' formula
        return self._flatten(x @ r_delta)

    def log(self, x, y):
        x = self._unflatten(x)
        y = self._unflatten(y)
        delta = x.transpose(-1, -2) @ y
        trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
        cos_theta = ((trace - 1) / 2).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.acos(cos_theta)[..., None]  # (..., n, 1, 1)
        diff = delta - delta.transpose(-1, -2)  # = 2 sin(theta) * hat(axis)
        sin_theta = torch.sin(theta).clamp_min(self.eps)
        a = torch.where(theta < self.eps, torch.zeros_like(diff), (theta / (2 * sin_theta)) * diff)
        return self._flatten(x @ a)


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
