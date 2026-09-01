"""Forecasting-error-vs-horizon — a world-model-appropriate evaluation.

The linear probe asks "is the representation linearly classifiable"; for a *world model*
the more honest question is "does it predict the future of the audio, and better than
something trivial". This module measures exactly that, on held-out audio, at each
horizon ``k``.

Two design rules, both learned the hard way:

**One space for everyone.** Model forecasts are converted to *raw* codec
(``reconstruct_raw``, using the statistics recorded on the checkpoint) and only then
standardized, with the same statistics applied to predictions and targets alike. The
earlier version standardized the target with the eval set's statistics while the
grounding head emitted into its *training* set's space, so on a transfer set the whole
mismatch was charged to the model — enough to flip the reported sign from +0.08 to −0.07
and manufacture a "the decoder doesn't transfer" finding that was not there.

**Persistence is not the bar.** ``x[t+k] := x[t]`` uses one frame and no fitting, and
codec embeddings are locally smooth, so beating it demonstrates very little: on ESC-50 a
ridge AR(4) reproduces the causal JEPA's entire gain over persistence. The headline
reference is therefore ``LinearAR`` (fit on the *training* distribution, so transfer
evaluation stays matched); persistence is kept only as a floor-of-the-floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import resolve_device
from ..data.stats import dataset_stats


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean per-frame cosine similarity between ``[1, M, D]`` tensors."""
    return F.cosine_similarity(a, b, dim=-1).mean()


def _skill(model_err: float, ref_err: float) -> float:
    return 1 - model_err / ref_err if ref_err > 0 else float("nan")


def _resolve_frames(model, max_frames):
    """Clamp scoring to the model's trained context unless told otherwise.

    Scoring past it measures positional extrapolation, not forecasting skill: on a
    256-frame model the per-frame latent error jumps from 0.45 to 0.67 at exactly frame 256
    while persistence sits at 0.64, so a long clip drags a genuinely positive skill negative.
    ``max_frames=None`` follows the checkpoint's recorded window; an explicit value is
    honoured but warned about when it exceeds it.
    """
    ctx = getattr(model, "trained_context", None) if model is not None else None
    if max_frames is None:
        return ctx or 512
    if ctx and max_frames > ctx:
        import warnings
        warnings.warn(
            f"Scoring {max_frames} frames but the model was trained on {ctx}. Frames beyond "
            f"{ctx} are positional extrapolation and will understate real skill — pass "
            f"max_frames<={ctx}, or use windowed_predict.", RuntimeWarning, stacklevel=3)
    return max_frames


def _codec_space(model, dataset, device, stats):
    """Resolve the standardization used to score codec-space forecasts, and warn if the
    model has no recorded output space (a pre-stats checkpoint) — its forecasts are then
    being interpreted in a space it never emitted into."""
    if stats is None:
        stats = dataset_stats(dataset)
    mu, sd = (s.reshape(1, 1, -1).to(device) for s in stats)
    if model is not None and hasattr(model, "has_codec_stats") and not model.has_codec_stats:
        import warnings
        warnings.warn(
            "Model carries no codec statistics (pre-stats checkpoint): its grounding head "
            "emits into an unknown space and codec-space forecasts will be biased. Supply "
            "the training cache (--train-stats) so set_codec_stats can be called.",
            RuntimeWarning, stacklevel=3,
        )
    return mu, sd


@dataclass
class HorizonMetrics:
    n_frames: int = 0
    latent_pred_l1: float = 0.0
    latent_persist_l1: float = 0.0
    latent_ar_l1: float = 0.0
    codec_pred_cos: float = 0.0
    codec_persist_cos: float = 0.0
    codec_ar_cos: float = 0.0
    codec_pred_l1: float = 0.0
    codec_persist_l1: float = 0.0
    codec_ar_l1: float = 0.0
    has_ar: bool = False

    def finalize(self) -> dict:
        n = max(1, self.n_frames)
        lp, lper, lar = (v / n for v in (self.latent_pred_l1, self.latent_persist_l1, self.latent_ar_l1))
        cp, cper, car = (v / n for v in (self.codec_pred_cos, self.codec_persist_cos, self.codec_ar_cos))
        clp, clper, clar = (v / n for v in (self.codec_pred_l1, self.codec_persist_l1, self.codec_ar_l1))
        out = {
            "latent_pred_l1": lp,
            "latent_persist_l1": lper,
            "latent_skill": _skill(lp, lper),
            "codec_pred_cos": cp,
            "codec_persist_cos": cper,
            "codec_cos_gain": cp - cper,
            "codec_pred_l1": clp,
            "codec_persist_l1": clper,
            "codec_l1_skill": _skill(clp, clper),
        }
        if self.has_ar:
            out.update({
                "latent_ar_l1": lar,
                "latent_skill_vs_ar": _skill(lp, lar),
                "codec_ar_cos": car,
                "codec_cos_gain_vs_ar": cp - car,
                "codec_ar_l1": clar,
                "codec_l1_skill_vs_ar": _skill(clp, clar),
            })
        return out


@torch.no_grad()
def codec_forecast_curves(
    dataset: Dataset,
    device: str | None = None,
    jepa: torch.nn.Module | None = None,
    apc: torch.nn.Module | None = None,
    ar: torch.nn.Module | None = None,
    max_clips: int | None = None,
    max_frames: int | None = None,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, dict]:
    """Codec-space forecasting curves for every predictor on the same axes.

    All predictions are produced in **raw codec space** and then standardized with one
    shared ``(mean, std)``, so persistence, ``LinearAR``, APC and the JEPA are directly
    comparable and none of them is charged for a normalization mismatch. Each model is
    evaluated at its own trained offsets. Returns ``{name: {offset: {cos, l1}}}``.
    """
    device = device or resolve_device("auto")
    max_frames = _resolve_frames(jepa, max_frames)
    mu, sd = _codec_space(jepa, dataset, device, stats)
    jepa = jepa.to(device).eval() if jepa is not None else None
    apc = apc.to(device).eval() if apc is not None else None
    ar = ar.to(device).eval() if ar is not None else None
    jepa_off = tuple(jepa.offsets) if jepa is not None else ()
    apc_off = tuple(apc.offsets) if apc is not None else ()
    ar_off = tuple(ar.offsets) if ar is not None else ()
    union = sorted(set(jepa_off) | set(apc_off) | set(ar_off))

    def fresh(offs):
        return {k: [0.0, 0.0, 0] for k in offs}

    P, A, J, R = fresh(union), fresh(apc_off), fresh(jepa_off), fresh(ar_off)

    def add(slot, pred_raw, tgt_std):
        """Standardize the raw prediction into the shared space, then score."""
        pred = (pred_raw - mu) / sd
        n = tgt_std.shape[1]
        slot[0] += float(_cos(pred, tgt_std)) * n
        slot[1] += float((pred - tgt_std).abs().mean()) * n
        slot[2] += n

    n = len(dataset) if max_clips is None else min(max_clips, len(dataset))
    for i in range(n):
        x = dataset[i]["features"][:max_frames].unsqueeze(0).to(device)
        T = x.shape[1]
        x_std = (x - mu) / sd
        for k in union:
            if T > k:
                add(P[k], x[:, :-k], x_std[:, k:])
        if ar is not None:
            preds = ar(x)
            for k in ar_off:
                if T > k:
                    add(R[k], preds[k][:, :-k], x_std[:, k:])
        if apc is not None:
            preds = apc(x)[0]
            for k in apc_off:
                if T > k:
                    add(A[k], preds[k][:, : T - k], x_std[:, k:])
        if jepa is not None:
            _, pr = jepa(x)
            for k in jepa_off:
                if T > k:
                    add(J[k], jepa.reconstruct_raw(pr[k][:, : T - k]), x_std[:, k:])

    def fin(acc):
        return {k: {"cos": s[0] / s[2], "l1": s[1] / s[2]} for k, s in acc.items() if s[2] > 0}

    out = {"persistence": fin(P)}
    if ar is not None:
        out[f"AR({ar.order})"] = fin(R)
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
    max_frames: int | None = None,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    codec_ar: torch.nn.Module | None = None,
    latent_ar: torch.nn.Module | None = None,
) -> dict[int, dict]:
    """Run the forecasting eval; returns ``{offset: finalized-metrics-dict}``.

    ``codec_ar`` / ``latent_ar`` are fitted ``LinearAR`` references (see
    ``fit_latent_ar``). When supplied, skill is reported against them as well as against
    persistence — and the AR figure is the one that means something.
    """
    device = device or resolve_device("auto")
    max_frames = _resolve_frames(jepa, max_frames)
    jepa = jepa.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    mu, sd = _codec_space(jepa, dataset, device, stats)
    codec_ar = codec_ar.to(device).eval() if codec_ar is not None else None
    latent_ar = latent_ar.to(device).eval() if latent_ar is not None else None
    offsets = tuple(jepa.offsets)
    acc = {k: HorizonMetrics(has_ar=codec_ar is not None or latent_ar is not None) for k in offsets}

    n = len(dataset) if max_clips is None else min(max_clips, len(dataset))
    for i in range(n):
        x = dataset[i]["features"][:max_frames].unsqueeze(0).to(device)  # [1, T, D]
        if x.shape[1] <= max(offsets) + 1:
            continue
        z, preds = jepa(x)
        z_tgt = target_encoder(x)
        x_std = (x - mu) / sd
        car = codec_ar(x) if codec_ar is not None else None
        lar = latent_ar(z_tgt) if latent_ar is not None else None
        for k in offsets:
            t = x.shape[1] - k
            m = acc[k]
            # latent space (model's own prediction vs latent-persistence / latent AR)
            m.latent_pred_l1 += float((preds[k][:, :t] - z_tgt[:, k:]).abs().mean()) * t
            m.latent_persist_l1 += float((z_tgt[:, k:] - z_tgt[:, :t]).abs().mean()) * t
            if lar is not None and k in lar:
                m.latent_ar_l1 += float((lar[k][:, :t] - z_tgt[:, k:]).abs().mean()) * t
            # codec space: every predictor produced in raw codec, standardized alike
            xhat = (jepa.reconstruct_raw(preds[k][:, :t]) - mu) / sd
            xfut = x_std[:, k:]
            m.codec_pred_cos += float(_cos(xhat, xfut)) * t
            m.codec_persist_cos += float(_cos(x_std[:, :t], xfut)) * t
            m.codec_pred_l1 += float((xhat - xfut).abs().mean()) * t
            m.codec_persist_l1 += float((x_std[:, :t] - xfut).abs().mean()) * t
            if car is not None and k in car:
                arhat = (car[k][:, :t] - mu) / sd
                m.codec_ar_cos += float(_cos(arhat, xfut)) * t
                m.codec_ar_l1 += float((arhat - xfut).abs().mean()) * t
            m.n_frames += t

    return {k: acc[k].finalize() for k in offsets}


@torch.no_grad()
def fit_latent_ar(
    encoder: torch.nn.Module,
    dataset: Dataset,
    offsets,
    order: int = 4,
    dim: int | None = None,
    max_clips: int = 200,
    max_frames: int | None = None,
    device: str | None = None,
):
    """Fit a ``LinearAR`` in an encoder's *latent* space — the trivial reference for the
    model's own latent-prediction skill, which latent-persistence badly under-states.

    Clips are strided across the dataset so a multi-cache (multi-domain) corpus contributes
    every domain; taking the first ``max_clips`` would draw them all from one.
    """
    from ..models.linear_ar import LinearAR, _stride

    device = device or resolve_device("auto")
    max_frames = max_frames or 512
    encoder = encoder.to(device).eval()
    seqs = []
    for i in _stride(len(dataset), max_clips):
        x = dataset[i]["features"][:max_frames].unsqueeze(0).to(device)
        seqs.append(encoder(x).cpu())
    d = dim or seqs[0].shape[-1]
    return LinearAR(d, order=order, offsets=offsets).fit(seqs)
