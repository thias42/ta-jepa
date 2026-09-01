"""Counterfactual evaluation of action conditioning (Phase 2).

This is the eval paired data buys, and the one descriptors could never support. Earlier
Phase 2 work perturbed a control, rendered audio, re-extracted a descriptor and read a
column-normalised matrix — so "does the dial work" was entangled with whether the descriptor
survived the codec and the render path. Here the intervention was *applied*, so the true
future is known and the question becomes a direct counterfactual:

    predict the future twice from the same context — once told the action that was actually
    applied, once told nothing — and score both against the true intervened future.

``action_gain`` = ``1 - err(with action) / err(without)``. Positive means the model used the
action; ~0 means it ignored it, which is what a dead dial looks like and what the Phase 2a
onset control did. Because the two predictions share a context and a target, content
difficulty cancels — the only difference is whether the action was supplied.

Scored **at the transition** (the frames the action actually spans), since away from an event
the action is zero by construction and both predictions are trivially identical.

Results are also **stratified by observability** — how transient the clean audio is, measured
as mean frame-to-frame codec flux. Reverb is the motivating case: a room is heard through its
response to transients, and the applied RT60 is recoverable at R²≈+0.22 on transient-rich
clips versus −2.0 on stationary ones. The stratification does double duty. It shows whether a
weak average is really a weak dial or just clips where the intervention is inaudible; and it
tests the stated risk of this whole design — if an axis works just as well on stationary
clips *where its effect should be near-inaudible*, the model is reading the applied DSP
rather than the acoustics.

**The expected ratio is per-axis, not universal.** A gain change is equally audible on
stationary and transient content, so gain *should* score about the same on both halves and a
ratio near 1.0 means nothing is wrong. The asymmetry test applies to **reverb**, whose
audibility genuinely depends on transients (and partly to tilt, on narrowband material).
Reading a ratio of 1.0 as evidence of DSP-detection for gain would be a misuse of this table.
"""

from __future__ import annotations

import torch

from ..config import resolve_device
from ..models.action_conditioning import action_windows


@torch.no_grad()
def counterfactual_report(
    model: torch.nn.Module,
    target_encoder: torch.nn.Module,
    dataset,
    offsets=None,
    n_clips: int = 200,
    device: str | None = None,
    axis_names: tuple[str, ...] = ("gain_db", "tilt_oct", "rt60_s"),
    horizon_pad: int = 4,
) -> dict:
    """Returns overall and per-axis ``action_gain`` plus the raw errors behind it.

    Per-axis figures use clips where **only** that axis fired, so an axis cannot borrow
    credit from a co-occurring one — the confound that made the Phase 2a matrices hard to
    read.
    """
    device = device or resolve_device("auto")
    model = model.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    offsets = tuple(offsets or model.offsets)

    records: list[dict] = []
    n = min(n_clips, len(dataset))
    for i in range(n):
        item = dataset[i]
        x = item["intervened"].unsqueeze(0).to(device)     # what the agent actually observes
        a = item["action"].unsqueeze(0).to(device)
        fired = (a.abs().sum(dim=(0, 1)) > 0)
        if not fired.any():
            continue
        span = torch.nonzero(a.abs().sum(-1)[0]).flatten()
        lo, hi = int(span[0]), int(span[-1])
        # observability: how transient the CLEAN audio is (the pre-flight's split variable)
        clean = item.get("features", item["intervened"])
        flux = float(clean.diff(dim=0).abs().mean())
        z_tgt = target_encoder(x)

        w_real = action_windows(a, offsets)
        w_null = {o: torch.zeros_like(v) for o, v in w_real.items()}
        _, p_real = model.predict_with_deltas(x, w_real)
        _, p_null = model.predict_with_deltas(x, w_null)

        only = axis_names[int(torch.nonzero(fired)[0])] if int(fired.sum()) == 1 else None
        for o in offsets:
            t = x.shape[1] - o
            s0, s1 = max(0, lo - o - horizon_pad), min(t, hi + horizon_pad)
            if s1 <= s0:
                continue
            tgt = z_tgt[:, o:][:, s0:s1]
            records.append({
                "offset": o, "axis": only, "flux": flux,
                "real": float((p_real[o][:, :t][:, s0:s1] - tgt).abs().mean()),
                "null": float((p_null[o][:, :t][:, s0:s1] - tgt).abs().mean()),
            })

    def gain(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        mr = sum(r["real"] for r in rows) / len(rows)
        mu = sum(r["null"] for r in rows) / len(rows)
        return {"n": len(rows), "err_with_action": mr, "err_without": mu,
                "action_gain": 1 - mr / mu if mu > 0 else float("nan")}

    out: dict = {"overall": {}, "per_axis": {}, "per_axis_by_observability": {}}
    for o in offsets:
        at_o = [r for r in records if r["offset"] == o]
        g = gain(at_o)
        if g:
            out["overall"][o] = g
        for nm in axis_names:
            rows = [r for r in at_o if r["axis"] == nm]
            g = gain(rows)
            if g:
                out["per_axis"].setdefault(nm, {})[o] = g
            # split within the axis, so the comparison is like-for-like
            if len(rows) >= 8:
                med = sorted(r["flux"] for r in rows)[len(rows) // 2]
                hi_g = gain([r for r in rows if r["flux"] > med])
                lo_g = gain([r for r in rows if r["flux"] <= med])
                if hi_g and lo_g:
                    out["per_axis_by_observability"].setdefault(nm, {})[o] = {
                        "transient": hi_g, "stationary": lo_g, "median_flux": med}
    return out
