"""Forecasting-error-vs-horizon eval — cross-model comparison curves.

Measures how well each model predicts the future of the audio (in codec space, vs a
persistence baseline) at each horizon — the world-model-appropriate counterpart to the
linear probe. Puts persistence (codec baseline), APC, and the causal JEPA on the same
axes. See src/tajepa/eval/forecasting.py.

    python scripts/run_forecast.py \
        --manifest data/manifests/esc50.jsonl --cache data/cache/encodec_24khz/esc50 \
        --split test --jepa-ckpt runs/jepa_fma_grounded.ckpt --apc-ckpt runs/apc_fma.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.embedding_dataset import ManifestEmbeddingDataset
from tajepa.eval import codec_forecast_curves, forecast_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--split", default=None, help="manifest split to eval on (default: all)")
    ap.add_argument("--jepa-ckpt", type=Path, default=None)
    ap.add_argument("--apc-ckpt", type=Path, default=None)
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
        jepa_lit = JEPALightning.load_from_checkpoint(str(args.jepa_ckpt), map_location="cpu")
        jepa = jepa_lit.jepa
    if args.apc_ckpt:
        from train_apc import APCLightning
        apc = APCLightning.load_from_checkpoint(str(args.apc_ckpt), map_location="cpu").model

    curves = codec_forecast_curves(ds, device=args.device, jepa=jepa, apc=apc,
                                   max_clips=args.max_clips)

    tag = f" [{args.split}]" if args.split else ""
    print(f"Codec-space forecasting on {args.manifest.name}{tag}  ({len(ds)} clips)")
    print("Cosine similarity of predicted vs true future frame; gain = over persistence.\n")
    persist = curves["persistence"]
    offs = sorted(persist)
    header = f"{'k':>3} | {'persistence':>11}"
    for name in ("APC", "JEPA"):
        if name in curves:
            header += f" | {name + ' cos (gain)':>18}"
    print(header); print("-" * len(header))
    for k in offs:
        row = f"{k:>3} | {persist[k]['cos']:>11.3f}"
        for name in ("APC", "JEPA"):
            if name in curves:
                if k in curves[name]:
                    c = curves[name][k]["cos"]
                    row += f" | {c:>8.3f} ({c - persist[k]['cos']:>+6.3f})"
                else:
                    row += f" | {'—':>18}"
        print(row)

    if jepa_lit is not None:
        print("\nJEPA latent-space skill (own space, vs latent-persistence):")
        rep = forecast_report(jepa, jepa_lit.target, ds, device=args.device,
                              max_clips=args.max_clips)
        print("  " + "  ".join(f"k={k}: {m['latent_skill']:+.1%}" for k, m in rep.items()))


if __name__ == "__main__":
    main()
