"""Phase 1 — causal latent JEPA (the core model).

A causal world model in codec-embedding space:

- **Frame encoder** ``f_θ`` (`CausalTransformer`): causal-masked transformer over the
  codec-embedding sequence -> online latents ``z_{1..T}`` (position ``t`` attends only
  to ``≤ t``).
- **Target encoder** ``f_θ̄``: an EMA copy of ``f_θ`` (managed by the trainer), stop-grad,
  producing the prediction targets ``z̄``.
- **Causal predictor** ``g_φ`` (`CausalPredictor`): from ``z_{≤t}`` predicts the EMA-target
  latents at several future offsets ``z̄_{t+o}``.

Loss = smooth-L1 in latent space against the stop-grad EMA target **plus VICReg
variance + covariance on the online latents**. The VICReg term is mandatory, not
optional (design invariant #3): unlike APC — which regresses a grounded input frame and
*cannot* collapse — here the target is a moving EMA representation that can collapse to a
constant at zero loss. Variance/covariance regularization + EMA + stop-grad is what
prevents that. Collapse diagnostics are monitored every step on top.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pe(t: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(t, device=device).float()[:, None]
    i = torch.arange(0, dim, 2, device=device).float()[None, :]
    angle = pos / (10000 ** (i / dim))
    pe = torch.zeros(t, dim, device=device)
    pe[:, 0::2] = angle.sin()
    pe[:, 1::2] = angle.cos()
    return pe


def causal_mask(t: int, device) -> torch.Tensor:
    """Boolean attention mask ``[T, T]``; ``True`` above the diagonal = cannot attend
    ahead. Boolean (not additive float) so it matches the boolean key-padding mask and
    avoids the mismatched-mask-type deprecation in ``nn.TransformerEncoder``."""
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


def _encoder_stack(dim, depth, heads, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=4 * dim,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class CausalTransformer(nn.Module):
    """Input-projected, positionally-encoded causal transformer: ``[B,T,in_dim] -> [B,T,dim]``."""

    def __init__(self, in_dim, dim, depth, heads, dropout=0.0) -> None:
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Identity() if in_dim == dim else nn.Linear(in_dim, dim)
        self.blocks = _encoder_stack(dim, depth, heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pad_mask=None) -> torch.Tensor:
        t = x.shape[1]
        h = self.in_proj(x) + sinusoidal_pe(t, self.dim, x.device)
        h = self.blocks(h, mask=causal_mask(t, x.device), src_key_padding_mask=pad_mask)
        return self.norm(h)


class CausalPredictor(nn.Module):
    """Causal transformer over the online latents + one head per future offset."""

    def __init__(self, dim, depth, heads, offsets, dropout=0.0) -> None:
        super().__init__()
        self.offsets = tuple(offsets)
        self.blocks = _encoder_stack(dim, depth, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleDict({str(o): nn.Linear(dim, dim) for o in self.offsets})

    def forward(self, z, pad_mask=None) -> dict[int, torch.Tensor]:
        t = z.shape[1]
        h = self.blocks(z, mask=causal_mask(t, z.device), src_key_padding_mask=pad_mask)
        h = self.norm(h)
        return {o: self.heads[str(o)](h) for o in self.offsets}


class JEPA(nn.Module):
    def __init__(
        self, in_dim=128, dim=256, enc_depth=6, pred_depth=3, heads=4,
        offsets=(1, 2, 3, 4), dropout=0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.offsets = tuple(offsets)
        self.encoder = CausalTransformer(in_dim, dim, enc_depth, heads, dropout)
        self.predictor = CausalPredictor(dim, pred_depth, heads, self.offsets, dropout)

    def forward(self, x, pad_mask=None):
        z = self.encoder(x, pad_mask)            # online latents [B,T,dim]
        preds = self.predictor(z, pad_mask)      # offset -> [B,T,dim]
        return z, preds

    def encode(self, x, pad_mask=None) -> torch.Tensor:
        return self.encoder(x, pad_mask)


# --------------------------------------------------------------------------- #
# Loss: latent prediction + VICReg (variance + covariance)
# --------------------------------------------------------------------------- #
def _valid_rows(z, pad_mask):
    return z[~pad_mask] if pad_mask is not None else z.reshape(-1, z.shape[-1])


def vicreg_terms(z, pad_mask=None, gamma: float = 1.0):
    """VICReg variance (hinge to std>=gamma) and covariance (off-diag -> 0) on ``z``."""
    x = _valid_rows(z, pad_mask)
    if x.shape[0] < 2:
        zero = z.new_zeros(())
        return zero, zero
    std = (x.var(dim=0) + 1e-4).sqrt()
    var_loss = F.relu(gamma - std).mean()
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    d = x.shape[1]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss, cov_loss


def jepa_loss(
    preds: dict[int, torch.Tensor],
    z_online: torch.Tensor,
    z_target: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    var_coef: float = 1.0,
    cov_coef: float = 0.04,
):
    """Smooth-L1 latent prediction (vs stop-grad EMA target) + VICReg on online latents."""
    z_target = z_target.detach()
    pred_loss = z_online.new_zeros(())
    logs: dict[str, float] = {}
    n = 0
    for o, pred in preds.items():
        if z_online.shape[1] <= o:
            continue
        p = pred[:, :-o]
        tgt = z_target[:, o:]
        if pad_mask is not None:
            valid = (~pad_mask[:, o:]).unsqueeze(-1)
            l = (F.smooth_l1_loss(p, tgt, reduction="none") * valid).sum() / (
                valid.sum().clamp(min=1) * p.shape[-1])
        else:
            l = F.smooth_l1_loss(p, tgt)
        pred_loss = pred_loss + l
        logs[f"pred_l1_n{o}"] = float(l.detach())
        n += 1
    pred_loss = pred_loss / max(1, n)

    var_loss, cov_loss = vicreg_terms(z_online, pad_mask)
    total = pred_loss + var_coef * var_loss + cov_coef * cov_loss
    logs.update(
        loss=float(total.detach()), pred_loss=float(pred_loss.detach()),
        var_loss=float(var_loss.detach()), cov_loss=float(cov_loss.detach()),
    )
    return total, logs


@torch.no_grad()
def latent_persistence_l1(z_target, offset, pad_mask=None) -> float:
    """Persistence baseline in latent space: predict ``z̄_{t+o} := z̄_t``."""
    if z_target.shape[1] <= offset:
        return float("nan")
    l1 = (z_target[:, offset:] - z_target[:, :-offset]).abs()
    if pad_mask is not None:
        valid = (~pad_mask[:, offset:]).unsqueeze(-1)
        return float((l1 * valid).sum() / (valid.sum().clamp(min=1) * z_target.shape[-1]))
    return float(l1.mean())
