import torch
import numpy as np
from .manifold import Sphere

class DBDSB_Riemannian:
    def __init__(self, sig, num_steps, timesteps, shape_x, shape_y, first_coupling, mean_match=True, ot_sampler=None, eps=1e-4, manifold=None, loss_type="bridge_matching", n_div_probes=1, n_bridge_steps=None, r_imf_eps=None, gamma_min=None, gamma_max=None, symmetric_gamma=False, gamma_space='linspace', **kwargs):
        assert ot_sampler is None
        assert first_coupling in ("ind", "ref")
        assert loss_type in ("bridge_matching", "score_divergence", "log_bridge")
        if loss_type == "bridge_matching":
            assert mean_match, "no closed-form manifold drift target; bridge_matching only supports mean_match=True"
        elif loss_type == "score_divergence":
            assert first_coupling == "ref", "score_divergence has no trajectory-based analogue of first_coupling='ind'; it always seeds n=1 from the driftless reference process, so first_coupling must be 'ref'"
        self.loss_type = loss_type
        self.n_div_probes = n_div_probes
        self.n_bridge_steps = n_bridge_steps
        self.r_imf_eps = r_imf_eps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.symmetric_gamma = symmetric_gamma
        self.gamma_space = gamma_space
        self.device = timesteps.device

        self.sig = sig
        self.num_steps = num_steps
        self.timesteps = timesteps
        assert len(self.timesteps) == self.num_steps
        assert torch.allclose(self.timesteps.sum(), torch.tensor(self.T))
        assert (self.timesteps > 0).all()
        self.gammas = self.timesteps * self.sig**2

        self.d_x = shape_x
        self.d_y = shape_y

        self.first_coupling = first_coupling
        self.eps = eps

        self.mean_match = mean_match
        self.manifold = manifold if manifold is not None else Sphere()

    @property
    def T(self):
        return 1.

    @property
    def alpha(self):
        return 0.

    @torch.no_grad()
    def marginal_prob(self, x, t, fb):
        raise NotImplementedError("no closed-form manifold marginal; keep loss_scale=False and std_trick=False")

    @torch.no_grad()
    def record_langevin_seq(self, net, samples_x, init_samples_y, fb, sample=False, num_steps=None, **kwargs):
        if fb == 'b':
            gammas = torch.flip(self.gammas, (0,))
            timesteps = torch.flip(self.timesteps, (0,))
            t = torch.ones((samples_x.shape[0], 1), device=self.device)
            sign = -1.
        elif fb == 'f':
            gammas = self.gammas
            timesteps = self.timesteps
            t = torch.zeros((samples_x.shape[0], 1), device=self.device)
            sign = 1.

        x = samples_x
        N = x.shape[0]

        if num_steps is None:
            num_steps = self.num_steps
        else:
            timesteps = np.interp(np.arange(1, num_steps+1)/num_steps, np.arange(self.num_steps+1)/self.num_steps, [0, *np.cumsum(timesteps.cpu())])
            timesteps = torch.from_numpy(np.diff(timesteps, prepend=[0])).to(self.device)
            gammas = timesteps * self.sig**2

        x_tot = torch.Tensor(N, num_steps, *self.d_x).to(x.device)
        y_tot = None
        steps_expanded = torch.Tensor(N, num_steps, 1).to(x.device)

        drift_fn = self.get_drift_fn_pred(fb)
        # score_divergence/log_bridge nets are trained (via tangent_field) to output a tangent
        # vector directly, used as drift as-is; only bridge_matching's mean_match nets predict
        # an endpoint that needs the log-map conversion in drift_fn
        tangent_valued_net = self.loss_type in ("score_divergence", "log_bridge")

        for k in range(num_steps):
            gamma = gammas[k]
            timestep = timesteps[k]

            pred = net(x, init_samples_y, t)

            if tangent_valued_net:
                drift = self.manifold.proj_tangent(x, pred)
                z_bar = torch.randn_like(x)
                z = self.manifold.proj_tangent(x, z_bar)
                g2 = self.eps_scale_at(t) 
                w = drift * timestep + torch.sqrt(g2 * timestep) * z
                x = self.manifold.exp(x, w)
            elif sample and (k == num_steps - 1):
                x = pred
            else:
                drift = drift_fn(t, x, pred)  # tangent vector at x, pointing towards pred
                z_bar = torch.randn_like(x)
                z = self.manifold.proj_tangent(x, z_bar)
                w = drift * timestep + torch.sqrt(gamma) * z
                x = self.manifold.exp(x, w)

            x_tot[:, k, :] = x
            steps_expanded[:, k, :] = t
            t = t + sign * timestep

        if fb == 'b':
            assert torch.allclose(t, torch.zeros(1, device=self.device), atol=1e-4, rtol=1e-4), f"{t} != 0"
        else:
            assert torch.allclose(t, torch.ones(1, device=self.device) * self.T, atol=1e-4, rtol=1e-4), f"{t} != 1"

        return x_tot, y_tot, None, steps_expanded

    @torch.no_grad()
    def generate_new_dataset(self, x0, y0, x1, sample_fn, sample_direction, sample=False, num_steps=None):
        if sample_direction == 'f':
            zstart = x0
        else:
            zstart = x1
        zend = self.record_langevin_seq(sample_fn, zstart, y0, sample_direction, sample=sample, num_steps=num_steps)[0][:, -1]
        if sample_direction == 'f':
            z0, z1 = zstart, zend
        else:
            z0, z1 = zend, zstart
        return z0, y0, z1

    @torch.no_grad()
    def probability_flow_ode(self, net_f=None, net_b=None, y=None):
        raise NotImplementedError("ODE sampling needs a manifold-aware (retraction-based) integrator; not implemented")

    @staticmethod
    def _zero_drift_net(x, y, t):
        # predicting the current point as its own endpoint gives log_x(x) = 0, i.e. no drift:
        # this makes record_langevin_seq simulate plain (driftless) manifold Brownian motion,
        # which is Q^0 = P, the reference process used to seed the first IPF iteration
        return x

    @torch.no_grad()
    def get_train_tuple(self, x0, x1, fb='', first_it=False):
        if first_it and fb == 'b':
            z0 = x0
            if self.first_coupling == "ref":
                z1 = self.record_langevin_seq(self._zero_drift_net, z0, None, 'f', sample=False)[0][:, -1]
            elif self.first_coupling == "ind":
                z1 = x1
            else:
                raise NotImplementedError
        elif first_it and fb == 'f':
            assert self.first_coupling == "ind"
            z0, z1 = x0, x1
        else:
            z0, z1 = x0, x1

        t = torch.rand(z1.shape[0], device=self.device) * (1-2*self.eps) + self.eps
        t = self._reshape_t(t, z0)

        # geodesic interpolation between endpoints, then tangent-space noise around it
        # (approximates the manifold Brownian bridge; no closed form in general)
        z_geo = self.manifold.exp(z0, t * self.manifold.log(z0, z1))
        z = torch.randn_like(z_geo)
        z = self.manifold.proj_tangent(z_geo, z)
        z_t = self.manifold.exp(z_geo, self.sig * torch.sqrt(t*(1.-t)) * z)

        if fb == 'f':
            target = z1
        else:
            target = z0
        return z_t, t, target

    def _reshape_t(self, t, x):
        return t.view(t.shape[0], *([1] * (len(x.shape) - 1)))

    def drift_f(self, t, x, pred):
        t = self._reshape_t(t, x)
        return self.manifold.log(x, pred) / (self.T - t)

    def drift_b(self, t, x, pred):
        t = self._reshape_t(t, x)
        return self.manifold.log(x, pred) / t

    def get_drift_fn_net(self, net, fb, y=None):
        drift_fn_pred = self.get_drift_fn_pred(fb)
        def drift_fn(t, x):
            pred = net(x, y, t)
            return drift_fn_pred(t, x, pred)
        return drift_fn

    def get_drift_fn_pred(self, fb):
        def drift_fn(t, x, pred):
            if fb == 'f':
                return self.drift_f(t, x, pred)
            else:
                return self.drift_b(t, x, pred)
        return drift_fn

    def tangent_field(self, net, x, y, t):
        return self.manifold.proj_tangent(x, net(x, y, t))

    def hutchinson_divergence(self, field_fn, x, n_probes=None):
        """Stochastic estimate of the Riemannian (tangential) divergence of `field_fn` at x."""
        n_probes = self.n_div_probes if n_probes is None else n_probes
        x = x if x.requires_grad else x.detach().requires_grad_(True)
        div_est = torch.zeros(x.shape[0], 1, device=x.device)
        for _ in range(n_probes):
            eps = self.manifold.proj_tangent(x, torch.randn_like(x))
            b = field_fn(x)
            vjp = torch.autograd.grad((b * eps).sum(), x, create_graph=True, retain_graph=True)[0]
            div_est = div_est + (vjp * eps).sum(dim=-1, keepdim=True)
        return div_est / n_probes

    def ipf_loss(self, net_train, net_fixed, forward_or_backward, x0, y0=None, num_steps=None):
        fixed_direction = 'f' if forward_or_backward == 'b' else 'b'
        traj_net = self._zero_drift_net if net_fixed is None else net_fixed
        with torch.no_grad():
            x_traj, _, _, t_traj = self.record_langevin_seq(traj_net, x0, y0, fixed_direction, sample=False, num_steps=num_steps)

        total_loss = 0.
        for k in range(x_traj.shape[1]):
            x_k = x_traj[:, k].detach().requires_grad_(True)
            t_k = t_traj[:, k]
            g_sq = self.eps_scale_at(t_k) 

            with torch.no_grad():
                f_k = torch.zeros_like(x_k) if net_fixed is None else self.tangent_field(net_fixed, x_k, y0, t_k)

            def b_fn(x, t_k=t_k):
                return self.tangent_field(net_train, x, y0, t_k)

            b_k = b_fn(x_k)
            div_b = self.hutchinson_divergence(b_fn, x_k)  # (N, 1)

            step_loss = 0.5 * ((f_k + b_k) ** 2).sum(dim=-1, keepdim=True) + g_sq * div_b  # (N, 1)
            total_loss = total_loss + step_loss.mean()

        return total_loss / x_traj.shape[1]

    def _default_bridge_steps(self):
        return self.num_steps if self.n_bridge_steps is None else self.n_bridge_steps

    def g_sq(self, t):
        if self.gamma_min is None or self.gamma_max is None:
            raise ValueError("g_sq requires gamma_min/gamma_max (or pass an explicit eps_scale)")
        frac = 1.0 - abs(2.0 * t - 1.0) if self.symmetric_gamma else t
        if self.gamma_space == 'geomspace':
            return self.gamma_min * (self.gamma_max / self.gamma_min) ** frac
        return self.gamma_min + (self.gamma_max - self.gamma_min) * frac

    def eps_scale_at(self, t):
        return self.r_imf_eps if self.r_imf_eps is not None else self.g_sq(t)

    @torch.no_grad()
    def reciprocal_projection(self, x0, x1, L=None, eps_scale=None):
        """
        Returns (traj, t_grid): traj is (N, L+1, *shape_x), traj[:, 0]=x0, traj[:, -1]=x1
        (Doob-pinned)."""
        L = self._default_bridge_steps() if L is None else L
        h = 1.0 / L

        X = x0
        traj = torch.empty(x0.shape[0], L + 1, *x0.shape[1:], device=x0.device)
        traj[:, 0] = X
        t = 0.0
        for l in range(L):
            g2 = self.eps_scale_at(t) if eps_scale is None else eps_scale
            v = self.manifold.log(X, x1)
            xi = self.manifold.proj_tangent(X, torch.randn_like(X))
            w = (h / (1.0 - t)) * v + np.sqrt(g2 * h) * xi
            X = self.manifold.exp(X, w)
            t = t + h
            traj[:, l + 1] = X

        traj[:, -1] = x1  # snap endpoint (Doob pinning)
        t_grid = torch.linspace(0.0, 1.0, L + 1, device=x0.device)
        return traj, t_grid

    @torch.no_grad()
    def markov_projection_train_tuple(self, x0, x1, fb, L=None, eps_scale=None):
        """The Doob-pinned bridge from x0 to x1 has two associated drifts (mirroring
        drift_f/drift_b above): the forward drift log(x_t,x1)/(1-t), singular at the
        x1 end (t=1), and the backward (time-reversed) drift log(x_t,x0)/t, singular
        at the x0 end (t=0). Train net_f on the former, net_b on the latter -- both
        regress the SAME underlying bridge path, just its two different-direction
        drifts; using the forward formula for both (as before) trained net_b to flow
        toward x1, i.e. to stay put at its own sampling start point."""
        L = self._default_bridge_steps() if L is None else L

        traj, t_grid = self.reciprocal_projection(x0, x1, L=L, eps_scale=eps_scale)

        N = x0.shape[0]
        if fb == 'f':
            idx = torch.randint(0, L, (N,), device=x0.device)  # exclude the pinned endpoint (t=1): 0/0
        else:
            idx = torch.randint(1, L + 1, (N,), device=x0.device)  # exclude the pinned endpoint (t=0): 0/0
        x_t = traj[torch.arange(N, device=x0.device), idx]
        t = t_grid[idx].view(N, *([1] * (len(x0.shape) - 1)))

        if fb == 'f':
            u = (1.0 / (1.0 - t)) * self.manifold.log(x_t, x1)
        else:
            u = (1.0 / t) * self.manifold.log(x_t, x0)
        return x_t, t, u

    def log_bridge_loss(self, net, x0, x1, fb, y=None, L=None, eps_scale=None):
        x_t, t, u = self.markov_projection_train_tuple(x0, x1, fb, L=L, eps_scale=eps_scale)
        v = self.tangent_field(net, x_t, y, t)
        return ((v - u) ** 2).sum(dim=-1).mean()
