import torch
from torch import nn

from .layers import MLP
from .time_embedding import get_timestep_embedding

N_AATYPES = 21  # len(restypes_with_x), vendor/mdgen/mdgen/residue_constants.py -- matches MDGen's
                # own amino-acid embedding vocab (20 standard types + "X" unknown)


class ScoreNetworkProteinCond(torch.nn.Module):
    """Peptide-conditioned drift network g_theta(x_t, t, c): same MLP skeleton as
    basic_cond.py::ScoreNetwork, plus an additive per-residue amino-acid-type embedding `c`
    (MDGen's own conditioning recipe). `c` depends only on invariant sequence identity, so
    adding its embedding to the already-invariant x_encoder features (rather than touching the
    tangent-space output) preserves whatever equivariance property the unconditioned network has
    -- see bridge/data/tetrapeptide_tps_multi.py for where `y` (the raw aatype indices) comes
    from."""

    def __init__(self, encoder_layers=[128], temb_dim=64, decoder_layers=[512, 512, 512],
                 x_dim=64, n_residues=4, cond_emb_dim=16, temb_max_period=10000):
        super().__init__()
        self.temb_dim = temb_dim
        t_enc_dim = temb_dim * 2
        self.n_residues = n_residues
        self.locals = [encoder_layers, temb_dim, decoder_layers, x_dim, n_residues, cond_emb_dim, temb_max_period]

        self.net = MLP(2 * t_enc_dim,
                        layer_widths=decoder_layers + [x_dim],
                        activate_final=False,
                        activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(temb_dim,
                              layer_widths=encoder_layers + [t_enc_dim],
                              activate_final=True,
                              activation_fn=torch.nn.LeakyReLU())

        self.x_encoder = MLP(x_dim,
                              layer_widths=[enc_dim * 2 for enc_dim in encoder_layers] + [t_enc_dim],
                              activate_final=True,
                              activation_fn=torch.nn.LeakyReLU())

        self.aatype_embed = nn.Embedding(N_AATYPES, cond_emb_dim)
        self.y_encoder = MLP(n_residues * cond_emb_dim,
                              layer_widths=[enc_dim * 2 for enc_dim in encoder_layers] + [t_enc_dim],
                              activate_final=True,
                              activation_fn=torch.nn.LeakyReLU())

        self.temb_max_period = temb_max_period

    def forward(self, x, y, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)

        t_emb = get_timestep_embedding(t, self.temb_dim, self.temb_max_period)
        t_emb = self.t_encoder(t_emb)

        c_emb = self.aatype_embed(y.long()).reshape(y.shape[0], self.n_residues * self.aatype_embed.embedding_dim)
        x_emb = self.x_encoder(x) + self.y_encoder(c_emb)

        h = torch.cat([x_emb, t_emb], -1)
        out = self.net(h)
        return out
