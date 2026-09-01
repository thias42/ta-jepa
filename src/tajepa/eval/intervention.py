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

    acc: dict[str, list[float]] = {}
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
        z_tgt = target_encoder(x)

        w_real = action_windows(a, offsets)
        w_null = {o: torch.zeros_like(v) for o, v in w_real.items()}
        _, p_real = model.predict_with_deltas(x, w_real)
        _, p_null = model.predict_with_deltas(x, w_null)

        for o in offsets:
            t = x.shape[1] - o
            # frames whose prediction window overlaps the action
            s0, s1 = max(0, lo - o - horizon_pad), min(t, hi + horizon_pad)
            if s1 <= s0:
                continue
            tgt = z_tgt[:, o:][:, s0:s1]
            e_real = float((p_real[o][:, :t][:, s0:s1] - tgt).abs().mean())
            e_null = float((p_null[o][:, :t][:, s0:s1] - tgt).abs().mean())
            acc.setdefault(f"k{o}_real", []).append(e_real)
            acc.setdefault(f"k{o}_null", []).append(e_null)
            if int(fired.sum()) == 1:                       # clean per-axis attribution
                nm = axis_names[int(torch.nonzero(fired)[0])]
                acc.setdefault(f"{nm}_k{o}_real", []).append(e_real)
                acc.setdefault(f"{nm}_k{o}_null", []).append(e_null)

    def gain(tag: str) -> dict | None:
        r, u = acc.get(f"{tag}_real"), acc.get(f"{tag}_null")
        if not r:
            return None
        mr, mu = sum(r) / len(r), sum(u) / len(u)
        return {"n": len(r), "err_with_action": mr, "err_without": mu,
                "action_gain": 1 - mr / mu if mu > 0 else float("nan")}

    out: dict = {"overall": {}, "per_axis": {}}
    for o in offsets:
        g = gain(f"k{o}")
        if g:
            out["overall"][o] = g
        for nm in axis_names:
            g = gain(f"{nm}_k{o}")
            if g:
                out["per_axis"].setdefault(nm, {})[o] = g
    return out
