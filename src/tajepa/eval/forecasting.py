"""Forecasting-error-vs-horizon — a world-model-appropriate evaluation.

The linear probe asks "is the representation linearly classifiable"; for a *world
model* the more honest question is "does it predict the future of the audio, better
than assuming nothing changes". This module measures exactly that.

For each horizon ``k`` (the model's trained offsets) it reports, on held-out audio:

- **codec-space skill** — decode the predicted latent back to codec space (via the
  grounding head) and compare to the true future codec frame. Primary metric is
  cosine similarity (robust to the decoder's standardization); standardized L1 is
  secondary. Skill = how much the model beats *persistence* ("predict x[t+k] := x[t]").
- **latent-space skill** — the model's own prediction error vs latent-persistence, for
  reference (this is what training optimizes).

Everything is relative to persistence, so a temporally-smooth latent — which makes the
absolute errors small for *both* the model and persistence — gets no free pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import resolve_device


def _per_clip_standardize(x: torch.Tensor) -> torch.Tensor:
    """``[1, T, D]`` z-scored per dimension over time."""
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(1e-4)
    return (x - mu) / sd


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean per-frame cosine similarity between ``[1, M, D]`` tensors."""
    return F.cosine_similarity(a, b, dim=-1).mean()


@dataclass
class HorizonMetrics:
    n_frames: int = 0
    latent_pred_l1: float = 0.0
    latent_persist_l1: float = 0.0
    codec_pred_cos: float = 0.0
    codec_persist_cos: float = 0.0
    codec_pred_l1: float = 0.0
    codec_persist_l1: float = 0.0

    def finalize(self) -> dict:
        n = max(1, self.n_frames)
        lp, lper = self.latent_pred_l1 / n, self.latent_persist_l1 / n
        cp, cper = self.codec_pred_cos / n, self.codec_persist_cos / n
        clp, clper = self.codec_pred_l1 / n, self.codec_persist_l1 / n
        return {
            "latent_pred_l1": lp,
            "latent_persist_l1": lper,
            "latent_skill": 1 - lp / lper if lper > 0 else float("nan"),
            "codec_pred_cos": cp,
            "codec_persist_cos": cper,
            "codec_cos_gain": cp - cper,
            "codec_pred_l1": clp,
            "codec_persist_l1": clper,
            "codec_l1_skill": 1 - clp / clper if clper > 0 else float("nan"),
        }


@torch.no_grad()
def _global_stats(dataset, device, n_clips, max_frames):
    """Per-dim mean/std of codec embeddings over a sample of clips (for a consistent,
    model-agnostic standardized space). Shapes ``[1, 1, D]``."""
    feats = [dataset[i]["features"][:max_frames] for i in range(min(n_clips, len(dataset)))]
    allf = torch.cat(feats, 0).to(device)
    return allf.mean(0)[None, None], allf.std(0).clamp_min(1e-4)[None, None]


@torch.no_grad()
def codec_forecast_curves(
    dataset: Dataset,
    device: str | None = None,
    jepa: torch.nn.Module | None = None,
    apc: torch.nn.Module | None = None,
    max_clips: int | None = None,
    max_frames: int = 512,
    stats_clips: int = 300,
) -> dict[str, dict]:
    """Codec-space forecasting curves for persistence / APC / JEPA on the same axes.

    All predictions are compared to the true future codec frame in one globally-
    standardized space, so the three are directly comparable. APC predicts codec
    frames directly (the strong reference); JEPA decodes its predicted latent via the
    grounding head; persistence is the codec baseline ("next = current"). Each model is
    evaluated at its own trained offsets. Returns ``{name: {offset: {cos, l1}}}`` with a
    ``"persistence"`` entry covering the union of offsets.
    """
    device = device or resolve_device("auto")
    mu, sd = _global_stats(dataset, device, stats_clips, max_frames)
    jepa = jepa.to(device).eval() if jepa is not None else None
    apc = apc.to(device).eval() if apc is not None else None
    jepa_off = tuple(jepa.offsets) if jepa is not None else ()
    apc_off = tuple(apc.offsets) if apc is not None else ()
    union = sorted(set(jepa_off) | set(apc_off))

    def fresh(offs):
        return {k: [0.0, 0.0, 0] for k in offs}

    P, A, J = fresh(union), fresh(apc_off), fresh(jepa_off)

    def add(slot, pred, tgt):
        n = tgt.shape[1]
        slot[0] += float(_cos(pred, tgt)) * n
        slot[1] += float((pred - tgt).abs().mean()) * n
        slot[2] += n

    n = len(dataset) if max_clips is None else min(max_clips, len(dataset))
    for i in range(n):
        x = dataset[i]["features"][:max_frames].unsqueeze(0).to(device)
        T = x.shape[1]
        x_std = (x - mu) / sd
        for k in union:
            if T > k:
                add(P[k], x_std[:, :-k], x_std[:, k:])
        if apc is not None:
            preds = apc(x)[0]
            for k in apc_off:
                if T > k:
                    add(A[k], (preds[k][:, : T - k] - mu) / sd, x_std[:, k:])
        if jepa is not None:
            _, pr = jepa(x)
            for k in jepa_off:
                if T > k:
                    add(J[k], jepa.reconstruct(pr[k][:, : T - k]), x_std[:, k:])

    def fin(acc):
        return {k: {"cos": s[0] / s[2], "l1": s[1] / s[2]} for k, s in acc.items() if s[2] > 0}

    out = {"persistence": fin(P)}
    if apc is not None:
        out["APC"] = fin(A)
    if jepa is not None:
        out["JEPA"] = fin(J)
    return out


@torch.no_grad()
def forecast_report(
    jepa: torch.nn.Module,
    target_encoder: torch.nn.Module,
    dataset: Dataset,
    device: str | None = None,
    max_clips: int | None = None,
    max_frames: int = 512,
) -> dict[int, dict]:
    """Run the forecasting eval; returns ``{offset: finalized-metrics-dict}``.

    ``jepa`` supplies ``encode`` / ``predictor`` (via ``forward``) / ``reconstruct``;
    ``target_encoder`` is the EMA encoder that defines the latent targets.
    """
    device = device or resolve_device("auto")
    jepa = jepa.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    offsets = tuple(jepa.offsets)
    acc = {k: HorizonMetrics() for k in offsets}

    n = len(dataset) if max_clips is None else min(max_clips, len(dataset))
    for i in range(n):
        x = dataset[i]["features"][:max_frames].unsqueeze(0).to(device)  # [1, T, D]
        if x.shape[1] <= max(offsets) + 1:
            continue
        z, preds = jepa(x)
        z_tgt = target_encoder(x)
        x_std = _per_clip_standardize(x)
        for k in offsets:
            t = x.shape[1] - k
            m = acc[k]
            # latent space (model's own prediction vs latent-persistence)
            m.latent_pred_l1 += float((preds[k][:, :t] - z_tgt[:, k:]).abs().mean()) * t
            m.latent_persist_l1 += float((z_tgt[:, k:] - z_tgt[:, :t]).abs().mean()) * t
            # codec space: decode predicted latent -> codec frame
            xhat = jepa.reconstruct(preds[k][:, :t])         # [1, t, D] standardized codec
            xfut = x_std[:, k:]
            m.codec_pred_cos += float(_cos(xhat, xfut)) * t
            m.codec_persist_cos += float(_cos(x_std[:, :t], xfut)) * t
            m.codec_pred_l1 += float((xhat - xfut).abs().mean()) * t
            m.codec_persist_l1 += float((x_std[:, :t] - xfut).abs().mean()) * t
            m.n_frames += t

    return {k: acc[k].finalize() for k in offsets}
