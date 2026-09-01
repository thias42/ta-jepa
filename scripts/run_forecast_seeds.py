"""Forecasting eval across several seeds — the figure with an error bar.

Every headline forecasting number in this project came from a single run, and the margin
over the linear-AR floor is small enough (+3-10%) that one run cannot distinguish it from
seed noise. This evaluates N checkpoints of the same config and reports **mean ± std**.

Two details that make the seeds comparable:

* The **codec** AR floor is model-independent, so it is fit once and shared. Every seed is
  scored against the same reference.
* The **latent** AR is not: each seed has its own latent space, so its reference must be fit
  in that space, per seed. Sharing one would compare seeds against a floor built for a
  different geometry.

    python scripts/run_forecast_seeds.py --manifest data/manifests/esc50.jsonl \\
        --cache data/cache/encodec_24khz/esc50 --train-cache <fma> <fsd50k> \\
        --jepa-ckpts runs/jepa_multi_v2_s*.ckpt
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.embedding_dataset import EmbeddingSequenceDataset, ManifestEmbeddingDataset
from tajepa.data.stats import codec_stats, ensure_codec_stats
from tajepa.eval import codec_forecast_curves, fit_latent_ar, forecast_report
from tajepa.models.linear_ar import fit_linear_ar


def _fmt(vals: list[float], pct: bool) -> str:
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:+.1%} ± {s:.1%}" if pct else f"{m:+.3f} ± {s:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--jepa-ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--train-cache", type=Path, nargs="+", default=None)
    ap.add_argument("--context-frames", type=int, default=None)
    ap.add_argument("--ar-order", type=int, default=4)
    ap.add_argument("--ar-clips", type=int, default=300)
    ap.add_argument("--max-clips", type=int, default=400)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_jepa import JEPALightning

    ds = ManifestEmbeddingDataset(args.manifest, args.cache)
    stats = codec_stats(args.train_cache) if args.train_cache else None
    fit_src = args.train_cache or [args.cache]

    codec_gain: dict[int, list[float]] = {}
    latent_skill: dict[int, list[float]] = {}
    ar = ar_ds = None

    for ck in args.jepa_ckpts:
        lit = _bootstrap.load_jepa_lightning(JEPALightning, ck)
        jepa = lit.jepa
        ensure_codec_stats(jepa, args.train_cache, what=f"JEPA {ck.name}")
        if args.context_frames and jepa.trained_context is None:
            jepa.set_context_frames(args.context_frames)
        ctx = jepa.trained_context or 512

        if ar is None:                      # codec AR is model-independent: fit once, share
            ar_ds = EmbeddingSequenceDataset(fit_src, window_frames=ctx, random_crop=False)
            ar = fit_linear_ar(ar_ds, dim=ds[0]["features"].shape[-1],
                               offsets=sorted(jepa.offsets), order=args.ar_order,
                               max_clips=args.ar_clips, max_frames=ctx)
            print(f"Fitted the shared codec AR({args.ar_order}) on {args.ar_clips} clips "
                  f"x {ctx} frames\n")

        curves = codec_forecast_curves(ds, device=args.device, jepa=jepa, ar=ar,
                                       max_clips=args.max_clips, stats=stats)
        # latent AR must be per-seed: each seed has its own latent geometry
        lat_ar = fit_latent_ar(lit.target, ar_ds, offsets=jepa.offsets, order=args.ar_order,
                               max_clips=min(200, args.ar_clips), max_frames=ctx,
                               device=args.device)
        rep = forecast_report(jepa, lit.target, ds, device=args.device,
                              max_clips=args.max_clips, stats=stats, latent_ar=lat_ar)

        arname = f"AR({args.ar_order})"
        line = []
        for k in sorted(curves["JEPA"]):
            g = curves["JEPA"][k]["cos"] - curves[arname][k]["cos"]
            codec_gain.setdefault(k, []).append(g)
            latent_skill.setdefault(k, []).append(rep[k]["latent_skill_vs_ar"])
            line.append(f"k={k}: {g:+.3f}")
        print(f"  {ck.stem:<24} codec vs AR  " + "  ".join(line))

    n = len(args.jepa_ckpts)
    print(f"\n{n} seeds — mean ± std (std is 0 with a single seed, which is the point)\n")
    print(f"{'k':>4}{'codec cos gain vs AR':>28}{'latent skill vs AR':>26}")
    for k in sorted(codec_gain):
        print(f"{k:>4}{_fmt(codec_gain[k], False):>28}{_fmt(latent_skill[k], True):>26}")

    if n > 1:
        worst = min(statistics.mean(v) - statistics.stdev(v) for v in codec_gain.values())
        print(f"\nWeakest horizon at mean−1σ (codec): {worst:+.3f} — "
              f"{'still positive' if worst > 0 else 'crosses zero, so the margin is not established'}")


if __name__ == "__main__":
    main()
