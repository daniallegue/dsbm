import torch
import torch.nn as nn
import numpy as np
from .optimal_transport import OTPlanSampler
from .manifold import Sphere

class DBDSB_Riemannian:
    def __init__(self, sig, num_steps, timesteps, shape_x, shape_y, first_coupling, mean_match=True, ot_sampler=None, eps=1e-4, manifold=None, loss_type="bridge_matching", n_div_probes=1, **kwargs):
        assert ot_sampler is None
        assert first_coupling in ("ind", "ref")
        assert loss_type in ("bridge_matching", "score_divergence")
        if loss_type == "bridge_matching":
            assert mean_match, "no closed-form manifold drift target; bridge_matching only supports mean_match=True"
        elif loss_type == "score_divergence":
            assert first_coupling == "ref", "score_divergence has no trajectory-based analogue of first_coupling='ind'; it always seeds n=1 from the driftless reference process, so first_coupling must be 'ref'"
        self.loss_type = loss_type
        self.n_div_probes = n_div_probes
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

        for k in range(num_steps):
            gamma = gammas[k]
            timestep = timesteps[k]

            pred = net(x, init_samples_y, t)  # predicted endpoint (mean_match=True)

            if sample and (k == num_steps - 1):
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

        g_sq = self.sig ** 2
        total_loss = 0.
        for k in range(x_traj.shape[1]):
            x_k = x_traj[:, k].detach().requires_grad_(True)
            t_k = t_traj[:, k]

            with torch.no_grad():
                f_k = torch.zeros_like(x_k) if net_fixed is None else self.tangent_field(net_fixed, x_k, y0, t_k)

            def b_fn(x, t_k=t_k):
                return self.tangent_field(net_train, x, y0, t_k)

            b_k = b_fn(x_k)
            div_b = self.hutchinson_divergence(b_fn, x_k)

            step_loss = 0.5 * ((f_k + b_k) ** 2).sum(dim=-1) + g_sq * div_b.squeeze(-1)
            total_loss = total_loss + step_loss.mean()

        return total_loss / x_traj.shape[1]
