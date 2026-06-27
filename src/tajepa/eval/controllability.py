"""Closed-loop controllability eval (Phase 2a).

The real test of supervised control (plan): perturb one control descriptor's delta,
*render the prediction back to audio*, re-run the MIR extractor, and check (a) the
intended attribute moved and (b) the others stayed put (disentanglement).

Render path: predicted latent → grounding head → un-standardize → codec decoder → audio
→ re-extract descriptors. We report the **controllability matrix** ``M[p, m]`` = mean
change in *measured* descriptor ``m`` when *perturbing* control ``p`` by ``+bump``. A
controllable, disentangled model has a positive, diagonally-dominant ``M``. Because we
measure the *difference* (perturbed − baseline), systematic render error (continuous-vs-
quantized decode, grounding loss) largely cancels — what matters is the induced change.

The render/measure steps are injected (``render_fn`` / ``desc_fn``) so the core is
testable offline; ``scripts/run_controllability.py`` wires the real EnCodec decoder +
descriptor frontend.
"""

from __future__ import annotations

import torch

from ..config import resolve_device


@torch.no_grad()
def controllability_matrix(
    model: torch.nn.Module,
    dataset,
    render_fn,                # std_codec [B,T,Dc] -> audio [B,1,N]
    desc_fn,                  # audio [B,1,N] -> descriptors [B,T,C]
    offset: int = 1,
    bump: float = 2.0,
    n_clips: int = 50,
    device: str | None = None,
    standardize_fn=None,      # raw control [B,T,C] -> standardized (as in training)
) -> tuple[torch.Tensor, int]:
    """Returns ``(M [C, C], n_used)`` — rows = perturbed control, cols = measured descriptor."""
    device = device or resolve_device("auto")
    model = model.eval().to(device)
    c = int(model.cond_dim)
    acc = torch.zeros(c, c)
    used = 0

    def render(deltas):
        # desc=ctrl supplies the (augmented) encoder input when the model uses it; steering
        # comes from the perturbed deltas. Ignored by non-augmented models.
        _, preds = model.predict_with_deltas(x, deltas, desc=ctrl)
        return desc_fn(render_fn(model.reconstruct(preds[offset]))).mean(dim=1)  # [B, C]

    n = min(n_clips, len(dataset))
    for idx in range(n):
        item = dataset[idx]
        x = item["features"].unsqueeze(0).to(device)
        ctrl = item["control"].unsqueeze(0).to(device)
        if standardize_fn is not None:
            ctrl = standardize_fn(ctrl)
        base = model.deltas_from(ctrl)
        base_desc = render(base)                       # [1, C]
        for p in range(c):
            bumped = {o: base[o].clone() for o in base}
            bumped[offset][..., p] += bump
            acc[p] += (render(bumped) - base_desc)[0].cpu()
        used += 1

    return acc / max(1, used), used


def disentanglement_report(M: torch.Tensor, names: list[str] | None = None) -> dict:
    """Summarize a controllability matrix.

    - ``diag``: targeted change (should be positive — bumping a control up raises its
      own measured descriptor).
    - ``diagonal_dominant``: fraction of controls whose largest-magnitude effect is on
      their own descriptor.
    - ``dominance_ratio``: mean |diagonal| / mean |off-diagonal|.
    """
    c = M.shape[0]
    diag = torch.diagonal(M)
    off = M - torch.diag(diag)
    dominant = sum(int(M[p].abs().argmax().item() == p) for p in range(c))
    off_mean = off.abs().sum() / max(1, c * c - c)
    return {
        "diag": [float(v) for v in diag],
        "diag_positive": [bool(v > 0) for v in diag],
        "diagonal_dominant_frac": dominant / c,
        "dominance_ratio": float(diag.abs().mean() / off_mean.clamp_min(1e-8)),
        "names": names,
    }
