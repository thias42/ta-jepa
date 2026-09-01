"""Forecasting-error-vs-horizon eval — cross-model comparison curves.

Measures how well each model predicts the future of the audio in codec space, at each
horizon, and puts every predictor on the same axes.

Two things this eval is careful about:

* **One standardization for everyone.** Model forecasts are produced in raw codec space
  (via the statistics recorded on the checkpoint) and only then standardized, with the
  same statistics applied to predictions and targets. Scoring a grounding head against
  the *eval* set's statistics while it emits into its *training* set's space charges the
  mismatch to the model, which on a transfer set is enough to flip the sign of the result.
* **A linear AR floor, not persistence.** ``x[t+k] := x[t]`` is trivially weak on locally
  smooth codec embeddings. The AR(p) reference is fit in closed form on the *training*
  cache — the same distribution the model saw — and is the number that matters.

    python scripts/run_forecast.py \
        --manifest data/manifests/esc50.jsonl --cache data/cache/encodec_24khz/esc50 \
        --split test --jepa-ckpt runs/jepa_fma_grounded.ckpt --apc-ckpt runs/apc_fma.ckpt \
        --train-cache data/cache/encodec_24khz/fma_small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.embedding_dataset import EmbeddingSequenceDataset, ManifestEmbeddingDataset
from tajepa.data.stats import codec_stats, ensure_codec_stats
from tajepa.eval import codec_forecast_curves, fit_latent_ar, forecast_report
from tajepa.models.linear_ar import fit_linear_ar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="Eval feature cache.")
    ap.add_argument("--split", default=None, help="manifest split to eval on (default: all)")
    ap.add_argument("--jepa-ckpt", type=Path, default=None)
    ap.add_argument("--apc-ckpt", type=Path, default=None)
    ap.add_argument("--train-cache", type=Path, nargs="+", default=None,
                    help="Cache the models were trained on. Fits the AR baseline there "
                         "(matched to the model) and supplies codec statistics for "
                         "pre-stats checkpoints.")
    ap.add_argument("--context-frames", type=int, default=None,
                    help="Sequence length the JEPA was trained on (its --window). Only "
                         "needed for pre-context checkpoints, which record none; without "
                         "it the eval scores past the training window and understates "
                         "skill, because absolute positional encodings do not extrapolate.")
    ap.add_argument("--ar-order", type=int, default=4,
                    help="Order of the linear-AR reference (0 disables it).")
    ap.add_argument("--ar-clips", type=int, default=300, help="Clips used to fit the AR.")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if not args.jepa_ckpt and not args.apc_ckpt:
        ap.error("provide at least one of --jepa-ckpt / --apc-ckpt")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ds = ManifestEmbeddingDataset(args.manifest, args.cache, split=args.split)

    jepa = jepa_lit = apc = None
    if args.jepa_ckpt:
        from train_jepa import JEPALightning
        jepa_lit = _bootstrap.load_jepa_lightning(JEPALightning, args.jepa_ckpt)
        jepa = jepa_lit.jepa
        ensure_codec_stats(jepa, args.train_cache, what=f"JEPA {args.jepa_ckpt.name}")
        if args.context_frames and jepa.trained_context is None:
            jepa.set_context_frames(args.context_frames)
            print(f"Context window set to {args.context_frames} frames "
                  f"(checkpoint recorded none)")
    if args.apc_ckpt:
        from train_apc import APCLightning
        apc = APCLightning.load_from_checkpoint(str(args.apc_ckpt), map_location="cpu").model

    # Score every predictor in the space the grounding head actually emits into.
    stats = codec_stats(args.train_cache) if args.train_cache else None

    # The AR floor, fit on the training distribution so transfer stays matched.
    ar = None
    if args.ar_order > 0:
        offsets = sorted(set(jepa.offsets if jepa else ()) | set(apc.offsets if apc else ()))
        fit_src = args.train_cache or [args.cache]
        # Fit inside the model's trained context: latents past it are extrapolation, and
        # fitting the reference on them measures the model's failure, not the audio.
        ctx = getattr(jepa, "trained_context", None) if jepa is not None else None
        ar_frames = ctx or 512
        ar_ds = EmbeddingSequenceDataset(fit_src, window_frames=ar_frames, random_crop=False)
        ar = fit_linear_ar(ar_ds, dim=ds[0]["features"].shape[-1], offsets=offsets,
                           order=args.ar_order, max_clips=args.ar_clips, max_frames=ar_frames)
        print(f"Fitted AR({args.ar_order}) on {min(args.ar_clips, len(ar_ds))} clips "
              f"x {ar_frames} frames from {', '.join(str(d) for d in fit_src)}"
              + (f"  (model context = {ctx})" if ctx else "  (model records no context)"))

    curves = codec_forecast_curves(ds, device=args.device, jepa=jepa, apc=apc, ar=ar,
                                   max_clips=args.max_clips, stats=stats)

    tag = f" [{args.split}]" if args.split else ""
    print(f"\nCodec-space forecasting on {args.manifest.name}{tag}  ({len(ds)} clips)")
    ref_name = f"AR({args.ar_order})" if ar is not None else "persistence"
    print(f"Cosine of predicted vs true future frame; gain is over **{ref_name}**.\n")

    ref = curves[ref_name]
    models = [n for n in ("persistence", f"AR({args.ar_order})", "APC", "JEPA") if n in curves]
    header = f"{'k':>3}" + "".join(f" | {n + ' cos (gain)':>20}" for n in models)
    print(header); print("-" * len(header))
    for k in sorted(curves[models[0]] if models[0] != "persistence" else curves["persistence"]):
        row = f"{k:>3}"
        for name in models:
            if k in curves[name]:
                c = curves[name][k]["cos"]
                g = f"{c - ref[k]['cos']:+.3f}" if k in ref else "  —  "
                row += f" | {c:>10.3f} ({g:>6})"
            else:
                row += f" | {'—':>20}"
        print(row)

    if jepa_lit is not None:
        lat_ar = None
        if ar is not None:
            lat_ar = fit_latent_ar(jepa_lit.target, ar_ds, offsets=jepa.offsets,
                                   order=args.ar_order, max_clips=min(200, args.ar_clips),
                                   max_frames=ar_frames, device=args.device)
        rep = forecast_report(jepa, jepa_lit.target, ds, device=args.device,
                              max_clips=args.max_clips, stats=stats,
                              codec_ar=ar, latent_ar=lat_ar)
        print("\nJEPA latent-space skill (own space):")
        print("  vs latent-persistence: " +
              "  ".join(f"k={k}: {m['latent_skill']:+.1%}" for k, m in rep.items()))
        if lat_ar is not None:
            print(f"  vs latent AR({args.ar_order}):     " +
                  "  ".join(f"k={k}: {m['latent_skill_vs_ar']:+.1%}" for k, m in rep.items()))


if __name__ == "__main__":
    main()
