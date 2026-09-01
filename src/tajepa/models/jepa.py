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


# --------------------------------------------------------------------------- #
# Grounding-head output space
# --------------------------------------------------------------------------- #
class CodecStatsMixin:
    """Owns the standardization the grounding head emits into.

    ``recon_head`` predicts a *standardized* codec frame, so its output only means
    something relative to a specific ``(mean, std)``. Those statistics are registered as
    buffers, so they are saved into the checkpoint and travel with the weights: training
    sets them once from the training cache, and every consumer (forecasting, rendering,
    demos) reads them back rather than inventing its own. Computing them ad hoc at the
    point of use is the bug this exists to prevent — evaluating the head's output against
    the *eval* set's statistics on a transfer set charges the mismatch to the model, which
    is enough to flip the sign of a forecasting result.

    Defaults are mean 0 / std 1 (i.e. "raw codec"), which is also what pre-stats
    checkpoints backfill to; ``set_codec_stats`` must be called for those to be correct.
    """

    def _init_codec_stats(self, in_dim: int) -> None:
        self.register_buffer("codec_mean", torch.zeros(in_dim))
        self.register_buffer("codec_std", torch.ones(in_dim))

    @torch.no_grad()
    def set_codec_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Record the codec standardization used for the grounding target."""
        self.codec_mean.copy_(mean.detach().reshape(-1).to(self.codec_mean))
        self.codec_std.copy_(std.detach().reshape(-1).to(self.codec_std).clamp_min(1e-4))

    @property
    def has_codec_stats(self) -> bool:
        """False when the buffers are still the identity (nothing was recorded)."""
        return bool((self.codec_std != 1).any() or (self.codec_mean != 0).any())

    def standardize_codec(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.codec_mean) / self.codec_std

    def reconstruct(self, z) -> torch.Tensor:
        """``z_t -> standardized codec frame`` (the grounding head's native output)."""
        return self.recon_head(z)

    def reconstruct_raw(self, z) -> torch.Tensor:
        """``z_t -> raw codec frame``. Use this whenever the output leaves the model —
        forecasting, rendering, decoding — so no caller has to guess the space."""
        return self.recon_head(z) * self.codec_std + self.codec_mean


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


class JEPA(CodecStatsMixin, nn.Module):
    def __init__(
        self, in_dim=128, dim=256, enc_depth=6, pred_depth=3, heads=4,
        offsets=(1, 2, 3, 4), dropout=0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.offsets = tuple(offsets)
        self.encoder = CausalTransformer(in_dim, dim, enc_depth, heads, dropout)
        self.predictor = CausalPredictor(dim, pred_depth, heads, self.offsets, dropout)
        # Grounding head: reconstruct the codec frame from the latent, to anchor the
        # latent to acoustically-rich content (optional; trainer weights it).
        self.recon_head = nn.Linear(dim, in_dim)
        self._init_codec_stats(in_dim)

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


def grounding_loss(recon, x, pad_mask=None, *, mean, std):
    """Masked MSE of the reconstructed codec frame against ``(x - mean) / std``.

    ``mean``/``std`` are **required and keyword-only** on purpose. They were previously
    computed per batch, which left the head emitting into an implicit, drifting space
    that no consumer could reconstruct — and evaluation code then standardized with
    whatever statistics were at hand. Pass fixed dataset statistics
    (``data.stats.codec_stats``) and record them on the model with ``set_codec_stats``.
    Standardizing still keeps this term O(1), so the grounding coefficient stays
    interpretable regardless of the raw codec embedding scale.
    """
    x_n = (x - mean) / std
    if pad_mask is not None:
        valid = (~pad_mask).unsqueeze(-1)
        return (F.mse_loss(recon, x_n, reduction="none") * valid).sum() / (
            valid.sum().clamp(min=1) * x.shape[-1])
    return F.mse_loss(recon, x_n)


def backfill_codec_stats(state_dict: dict, prefix: str, in_dim: int) -> dict:
    """Add identity ``codec_mean``/``codec_std`` to a pre-stats checkpoint.

    Lets older checkpoints load, but they carry *no* record of the space their grounding
    head emits into — callers must supply it (``--train-stats``) or codec-space numbers
    will be wrong in exactly the way this machinery exists to prevent.
    """
    for name, default in (("codec_mean", torch.zeros(in_dim)), ("codec_std", torch.ones(in_dim))):
        state_dict.setdefault(f"{prefix}{name}", default)
    return state_dict


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
