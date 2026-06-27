"""Collapse diagnostics.

Anti-collapse is a first-class, always-on concern (design invariant #3). These
metrics are wired into baseline training now so the exact same monitors carry into
Phase 1, where the EMA target genuinely *can* collapse to a constant at zero loss:

- ``feature_std``: mean per-dimension standard deviation. -> 0 under collapse.
- ``effective_rank``: entropy-based effective rank of the feature covariance
  (Roy & Vetterli). Drops toward 1 as representations concentrate on a subspace.
"""

from __future__ import annotations

import torch


def _flatten(feats: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """``[B, T, D]`` (+ optional ``[B, T]`` pad mask) -> valid rows ``[N, D]``."""
    if feats.dim() == 3:
        if pad_mask is not None:
            feats = feats[~pad_mask]
        else:
            feats = feats.reshape(-1, feats.shape[-1])
    return feats


@torch.no_grad()
def feature_std(feats: torch.Tensor, pad_mask: torch.Tensor | None = None) -> float:
    x = _flatten(feats, pad_mask)
    if x.shape[0] < 2:
        return float("nan")
    return float(x.std(dim=0).mean())


@torch.no_grad()
def effective_rank(feats: torch.Tensor, pad_mask: torch.Tensor | None = None) -> float:
    x = _flatten(feats, pad_mask)
    if x.shape[0] < 2:
        return float("nan")
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / (x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov.float()).clamp(min=0)
    s = eig.sum()
    if s <= 0:
        return float("nan")
    p = eig / s
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy))


@torch.no_grad()
def codebook_perplexity(indices: torch.Tensor, num_codes: int) -> float:
    """Perplexity of VQ code usage (1 = one code used, ``num_codes`` = uniform).

    The anti-collapse / leakage monitor for learned latent actions (plan, Phase 2b):
    too low = index collapse (a dead codebook); too high with trivial prediction = the
    action may be leaking the whole transition."""
    counts = torch.bincount(indices.reshape(-1), minlength=num_codes).float()
    p = counts / counts.sum().clamp_min(1)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def collapse_report(feats: torch.Tensor, pad_mask: torch.Tensor | None = None) -> dict[str, float]:
    return {
        "feature_std": feature_std(feats, pad_mask),
        "effective_rank": effective_rank(feats, pad_mask),
    }
