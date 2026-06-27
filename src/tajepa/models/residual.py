"""Phase 2a+2b — residual learned actions.

Pure 2b learned the codebook as a *loudness* axis (the dominant, easily-predicted
transition) — redundant with the 2a descriptors and missing the transient/texture control
we wanted. Fix: give the predictor the supervised descriptor delta *for free* (FiLM), and
keep the learned action on a *small* VQ codebook. With loudness/brightness already supplied
by the cheap descriptor path, the scarce code budget can't afford to re-encode them, so it
is pushed onto the **residual** transition (what descriptors don't explain).

So control becomes two complementary handles: descriptor deltas (loudness/brightness, the
known envelope axes) + learned action codes (the residual: transients/texture). The inverse
model also sees the descriptor delta, so it can factor out the descriptor-explained part.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .jepa import CausalTransformer, _encoder_stack, causal_mask
from .control import FiLM
from .actions import VectorQuantizer


def _next_step_delta(desc: torch.Tensor) -> torch.Tensor:
    """``desc [B,T,C]`` -> ``desc[t+1] - desc[t]`` (last frame padded with 0)."""
    future = torch.cat([desc[:, 1:], desc[:, -1:]], dim=1)
    return future - desc


class ResidualInverse(nn.Module):
    """Infer the action from ``(z_t, z_{t+1}, Δdesc_t)`` — can factor out the descriptor."""

    def __init__(self, dim: int, cond_dim: int, code_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 2 * dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim + cond_dim, hidden), nn.GELU(), nn.Linear(hidden, code_dim)
        )

    def forward(self, z: torch.Tensor, desc_delta: torch.Tensor) -> torch.Tensor:
        z_next = torch.cat([z[:, 1:], z[:, -1:]], dim=1)
        return self.net(torch.cat([z, z_next, desc_delta], dim=-1))


class ResidualPredictor(nn.Module):
    """Causal predictor of ``z_{t+1}``, FiLM-conditioned on descriptor delta THEN action."""

    def __init__(self, dim, depth, heads, cond_dim, code_dim, dropout=0.0) -> None:
        super().__init__()
        self.blocks = _encoder_stack(dim, depth, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.film_desc = FiLM(cond_dim, dim)
        self.film_act = FiLM(code_dim, dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, z, desc_delta, action, pad_mask=None) -> torch.Tensor:
        t = z.shape[1]
        h = self.norm(self.blocks(z, mask=causal_mask(t, z.device), src_key_padding_mask=pad_mask))
        h = self.film_desc(h, desc_delta)      # descriptors handle loudness/brightness (free)
        h = self.film_act(h, action)           # learned action handles the residual
        return self.head(h)


class ResidualActionJEPA(nn.Module):
    def __init__(self, in_dim=128, dim=256, enc_depth=6, pred_depth=3, heads=4,
                 cond_dim=3, num_codes=16, code_dim=32, commitment_cost=0.25, dropout=0.0) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.encoder = CausalTransformer(in_dim, dim, enc_depth, heads, dropout)
        self.inverse = ResidualInverse(dim, cond_dim, code_dim)
        self.vq = VectorQuantizer(num_codes, code_dim, commitment_cost)
        self.predictor = ResidualPredictor(dim, pred_depth, heads, cond_dim, code_dim, dropout)
        self.recon_head = nn.Linear(dim, in_dim)

    def forward(self, x, desc, pad_mask=None):
        z = self.encoder(x, pad_mask)
        dd = _next_step_delta(desc)
        e = self.inverse(z, dd)
        q, idx, vq_loss, probs = self.vq(e)
        pred = self.predictor(z, dd, q, pad_mask)
        return {"z": z, "pred": pred, "indices": idx, "vq_loss": vq_loss, "probs": probs}

    def predict_with(self, x, desc, codes, pad_mask=None):
        """Inference: drive with the descriptor control (``desc``) and chosen action ``codes``."""
        z = self.encoder(x, pad_mask)
        q = self.vq.embedding(codes)
        return z, self.predictor(z, _next_step_delta(desc), q, pad_mask)

    def reconstruct(self, z) -> torch.Tensor:
        return self.recon_head(z)

    def encode(self, x, pad_mask=None) -> torch.Tensor:
        return self.encoder(x, pad_mask)
