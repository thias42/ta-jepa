"""Actions-controllability eval (Phase 2b).

Learned actions have no labels, so we characterize what each code *does* in interpretable
(descriptor) terms. For every code ``k`` we force it at all positions, render the predicted
next latent back to audio, re-extract the MIR descriptors, and take the change vs the
baseline (inferred actions) as the code's **effect signature** ``e_k`` in descriptor space.

A good learned action vocabulary is:
- **consistent** — a code's effect is similar across clips (``within_std`` small relative to
  the effect magnitude), i.e. the code means the same thing in different contexts;
- **distinct** — different codes have different effects (``separability`` = between-code
  spread / within-code noise is high), i.e. the codebook isn't redundant;
- **used** — the inverse model actually assigns a spread of codes (``usage``).

Render/measure are injected so the core is testable offline; ``run_action_eval.py`` wires
the real EnCodec decoder + descriptor frontend.
"""

from __future__ import annotations

import torch

from ..config import resolve_device


@torch.no_grad()
def action_effect_matrix(
    model: torch.nn.Module,
    dataset,
    render_fn,                # std_codec [B,T,Dc] -> audio [B,1,N]
    desc_fn,                  # audio -> descriptors [B,T,C]
    n_clips: int = 40,
    device: str | None = None,
) -> dict:
    """Returns ``effect [K,C]``, ``within_std [K]``, ``usage [K]``."""
    device = device or resolve_device("auto")
    model = model.eval().to(device)
    k = int(model.num_codes)
    per_code: list[list[torch.Tensor]] = [[] for _ in range(k)]
    usage = torch.zeros(k)

    def desc_of(pred):
        return desc_fn(render_fn(model.reconstruct(pred))).mean(dim=1)[0].cpu()  # [C]

    n = min(n_clips, len(dataset))
    for idx in range(n):
        x = dataset[idx]["features"].unsqueeze(0).to(device)
        out = model(x)
        usage += torch.bincount(out["indices"].reshape(-1).cpu(), minlength=k).float()
        base = desc_of(out["pred"])
        t = x.shape[1]
        for code in range(k):
            codes = torch.full((1, t), code, dtype=torch.long, device=device)
            _, pred_c = model.predict_with_actions(x, codes)
            per_code[code].append(desc_of(pred_c) - base)

    effect = torch.stack([torch.stack(per_code[c]).mean(0) for c in range(k)])      # [K,C]
    within = torch.stack([torch.stack(per_code[c]).std(0).mean() for c in range(k)])  # [K]
    return {"effect": effect, "within_std": within, "usage": usage / usage.sum().clamp_min(1)}


@torch.no_grad()
def residual_action_effect_matrix(
    model: torch.nn.Module,
    dataset,                  # PairedSequenceDataset (features + control)
    render_fn,
    desc_fn,
    standardize_fn,           # raw control -> standardized (as in training)
    n_clips: int = 40,
    device: str | None = None,
) -> dict:
    """Same as ``action_effect_matrix`` but for the residual model — keeps the real
    descriptor control fixed and varies only the learned action code, so the effects
    isolate what the *codes* (not the descriptors) do."""
    device = device or resolve_device("auto")
    model = model.eval().to(device)
    k = int(model.num_codes)
    per_code: list[list[torch.Tensor]] = [[] for _ in range(k)]
    usage = torch.zeros(k)

    def desc_of(pred):
        return desc_fn(render_fn(model.reconstruct(pred))).mean(dim=1)[0].cpu()

    n = min(n_clips, len(dataset))
    for idx in range(n):
        item = dataset[idx]
        x = item["features"].unsqueeze(0).to(device)
        desc = standardize_fn(item["control"].unsqueeze(0).to(device))
        out = model(x, desc)
        usage += torch.bincount(out["indices"].reshape(-1).cpu(), minlength=k).float()
        base = desc_of(out["pred"])
        t = x.shape[1]
        for code in range(k):
            codes = torch.full((1, t), code, dtype=torch.long, device=device)
            _, pred_c = model.predict_with(x, desc, codes)
            per_code[code].append(desc_of(pred_c) - base)

    effect = torch.stack([torch.stack(per_code[c]).mean(0) for c in range(k)])
    within = torch.stack([torch.stack(per_code[c]).std(0).mean() for c in range(k)])
    return {"effect": effect, "within_std": within, "usage": usage / usage.sum().clamp_min(1)}


def action_report(res: dict, names: list[str] | None = None) -> dict:
    """Summarize an action-effect result into consistency / distinctiveness scores."""
    M, within = res["effect"], res["within_std"]
    mag = M.norm(dim=1)                                  # [K] effect magnitude per code
    consistency = mag / (mag + within + 1e-8)            # [K] in [0,1]; 1 = noiseless effect
    between = M.std(dim=0).mean()                        # spread across codes
    separability = float(between / (within.mean() + 1e-8))
    dominant = [int(M[c].abs().argmax()) for c in range(M.shape[0])]
    return {
        "consistency": [float(v) for v in consistency],
        "mean_consistency": float(consistency.mean()),
        "separability": separability,
        "dominant_descriptor": dominant,
        "usage": [float(v) for v in res["usage"]],
        "names": names,
    }
