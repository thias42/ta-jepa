"""Phase 2a — supervised control via descriptor-delta FiLM conditioning.

The controllable JEPA reuses the Phase 1 causal encoder ``f_θ`` and adds a predictor
whose per-offset heads are FiLM-modulated by the **delta** of frame-aligned MIR
descriptors — i.e. "how should loudness / brightness / onset / pitch change from ``t`` to
``t+o``". Conditioning on the *delta* (not absolute state) makes control a transition
modulation (plan, Phase 2a): at training the delta is the observed future change (so the
predictor learns to use it); at inference you set the delta to steer the prediction.

FiLM layers are zero-initialized so the model starts as an unconditioned JEPA and learns
to lean on the control — keeping it from short-cutting early. Descriptors are low-dim
(3–5) vs the latent (256), so they modulate rather than determine ``z``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .jepa import CausalTransformer, _encoder_stack, causal_mask


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``h -> (1 + gamma(c)) * h + beta(c)``.

    Zero-initialized -> identity at init (starts unconditioned)."""

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gb(c).chunk(2, dim=-1)
        return (1 + gamma) * h + beta


class ControllablePredictor(nn.Module):
    """Causal predictor with a FiLM-conditioned head per future offset."""

    def __init__(self, dim, depth, heads, offsets, cond_dim, dropout=0.0) -> None:
        super().__init__()
        self.offsets = tuple(offsets)
        self.blocks = _encoder_stack(dim, depth, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.films = nn.ModuleDict({str(o): FiLM(cond_dim, dim) for o in self.offsets})
        self.heads = nn.ModuleDict({str(o): nn.Linear(dim, dim) for o in self.offsets})

    def forward(self, z, deltas: dict[int, torch.Tensor], pad_mask=None) -> dict[int, torch.Tensor]:
        t = z.shape[1]
        h = self.norm(self.blocks(z, mask=causal_mask(t, z.device), src_key_padding_mask=pad_mask))
        return {o: self.heads[str(o)](self.films[str(o)](h, deltas[o])) for o in self.offsets}


class ControllableJEPA(nn.Module):
    def __init__(
        self, in_dim=128, dim=256, enc_depth=6, pred_depth=3, heads=4,
        offsets=(1, 2, 3, 4), cond_dim=3, dropout=0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.offsets = tuple(offsets)
        self.encoder = CausalTransformer(in_dim, dim, enc_depth, heads, dropout)
        self.predictor = ControllablePredictor(dim, pred_depth, heads, self.offsets, cond_dim, dropout)
        self.recon_head = nn.Linear(dim, in_dim)   # grounding / latent->codec decoder

    def deltas_from(self, desc: torch.Tensor) -> dict[int, torch.Tensor]:
        """``desc [B,T,C]`` -> ``{offset: desc[t+o] - desc[t]}`` (last frames padded with 0)."""
        out = {}
        for o in self.offsets:
            future = torch.cat([desc[:, o:], desc[:, -1:].expand(-1, o, -1)], dim=1)
            out[o] = future - desc
        return out

    def forward(self, x, desc, pad_mask=None):
        z = self.encoder(x, pad_mask)
        preds = self.predictor(z, self.deltas_from(desc), pad_mask)
        return z, preds

    def predict_with_deltas(self, x, deltas: dict[int, torch.Tensor], pad_mask=None):
        """Inference: predict using *chosen* per-offset descriptor deltas (steering)."""
        z = self.encoder(x, pad_mask)
        return z, self.predictor(z, deltas, pad_mask)

    def reconstruct(self, z) -> torch.Tensor:
        return self.recon_head(z)

    def encode(self, x, pad_mask=None) -> torch.Tensor:
        return self.encoder(x, pad_mask)
