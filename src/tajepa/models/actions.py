"""Phase 2b — learned latent actions (LAPO / Genie-style).

Some control axes (transients, texture, polyphony — the dead onset dial of 2a) aren't
captured by hand-defined descriptors. So we *learn* a discrete action vocabulary:

- **Inverse model** ``q(a_t | z_t, z_{t+1})``: infers, from the current and next latent,
  a continuous action that explains the transition.
- **VQ bottleneck**: quantizes the action to a *small* codebook. The small codebook is the
  key anti-leakage device — a few bits per step can't encode the whole future, so the code
  must capture the salient, reusable part of the transition.
- **Forward predictor** ``g(z_{≤t}, a_t) -> z_{t+1}``: causal in ``z``, FiLM-conditioned on
  the action. At inference the inverse model is dropped and you drive with chosen codes.

Risk is shortcut/leak (the action smuggling the answer); mitigate with a deliberately small
codebook, the VQ commitment loss, and (in the trainer) a code-usage entropy bonus. Per the
plan, do NOT select on prediction loss — watch codebook perplexity + downstream control.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jepa import CausalTransformer, _encoder_stack, causal_mask
from .control import FiLM


class VectorQuantizer(nn.Module):
    """Straight-through VQ with codebook + commitment loss."""

    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.embedding = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)
        self.commitment_cost = commitment_cost

    def forward(self, e: torch.Tensor):
        """``e [B,T,d]`` -> ``(quantized_st [B,T,d], indices [B,T], vq_loss, probs [B,T,K])``.

        ``probs`` is the soft (softmax-over-negative-distance) assignment — used by the
        trainer for a differentiable code-usage entropy bonus."""
        flat = e.reshape(-1, e.shape[-1])
        dist = torch.cdist(flat, self.embedding.weight)        # [N, K]
        idx = dist.argmin(dim=1)
        q = self.embedding(idx).view_as(e)
        codebook_loss = F.mse_loss(q, e.detach())
        commit_loss = F.mse_loss(e, q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commit_loss
        q_st = e + (q - e).detach()                            # straight-through estimator
        probs = F.softmax(-dist, dim=1).view(*e.shape[:-1], self.num_codes)
        return q_st, idx.view(e.shape[:-1]), vq_loss, probs


class InverseModel(nn.Module):
    """Infers the action embedding from ``(z_t, z_{t+1})``."""

    def __init__(self, dim: int, code_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 2 * dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, code_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_next = torch.cat([z[:, 1:], z[:, -1:]], dim=1)       # last frame padded
        return self.net(torch.cat([z, z_next], dim=-1))        # [B, T, code_dim]


class ActionPredictor(nn.Module):
    """Causal predictor of ``z_{t+1}``, FiLM-conditioned on the action code."""

    def __init__(self, dim, depth, heads, code_dim, dropout=0.0) -> None:
        super().__init__()
        self.blocks = _encoder_stack(dim, depth, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.film = FiLM(code_dim, dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, z, action, pad_mask=None) -> torch.Tensor:
        t = z.shape[1]
        h = self.norm(self.blocks(z, mask=causal_mask(t, z.device), src_key_padding_mask=pad_mask))
        return self.head(self.film(h, action))


class ActionJEPA(nn.Module):
    def __init__(self, in_dim=128, dim=256, enc_depth=6, pred_depth=3, heads=4,
                 num_codes=16, code_dim=32, commitment_cost=0.25, dropout=0.0) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.encoder = CausalTransformer(in_dim, dim, enc_depth, heads, dropout)
        self.inverse = InverseModel(dim, code_dim)
        self.vq = VectorQuantizer(num_codes, code_dim, commitment_cost)
        self.predictor = ActionPredictor(dim, pred_depth, heads, code_dim, dropout)
        self.recon_head = nn.Linear(dim, in_dim)               # grounding / latent->codec

    def forward(self, x, pad_mask=None):
        z = self.encoder(x, pad_mask)
        e = self.inverse(z)
        q, idx, vq_loss, probs = self.vq(e)
        pred = self.predictor(z, q, pad_mask)                  # predicts z_{t+1}
        return {"z": z, "pred": pred, "indices": idx, "vq_loss": vq_loss, "probs": probs}

    def predict_with_actions(self, x, codes, pad_mask=None):
        """Inference steering: predict ``z_{t+1}`` from chosen action ``codes [B,T]``."""
        z = self.encoder(x, pad_mask)
        q = self.vq.embedding(codes)
        return z, self.predictor(z, q, pad_mask)

    def reconstruct(self, z) -> torch.Tensor:
        return self.recon_head(z)

    def encode(self, x, pad_mask=None) -> torch.Tensor:
        return self.encoder(x, pad_mask)
