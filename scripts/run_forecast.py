"""Forecasting-error-vs-horizon eval for a causal JEPA checkpoint.

Measures how well the model predicts the future of the audio (in codec space, vs a
persistence baseline) at each horizon — the world-model-appropriate counterpart to the
linear probe. See src/tajepa/eval/forecasting.py.

    python scripts/run_forecast.py --jepa-ckpt runs/jepa_fma_grounded.ckpt \
        --manifest data/manifests/esc50.jsonl --cache data/cache/encodec_24khz/esc50 \
        --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.embedding_dataset import ManifestEmbeddingDataset
from tajepa.eval import forecast_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jepa-ckpt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--split", default=None, help="manifest split to eval on (default: all)")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_jepa import JEPALightning

    lit = JEPALightning.load_from_checkpoint(str(args.jepa_ckpt), map_location="cpu")
    ds = ManifestEmbeddingDataset(args.manifest, args.cache, split=args.split)
    report = forecast_report(lit.jepa, lit.target, ds, device=args.device, max_clips=args.max_clips)

    print(f"Forecasting: {args.jepa_ckpt.name} on {args.manifest.name}"
          f"{f' [{args.split}]' if args.split else ''}  ({len(ds)} clips)")
    print(f"{'k':>3} | {'codec cos (pred/persist/gain)':^34} | {'codec L1 skill':>14} | {'latent skill':>12}")
    print("-" * 74)
    for k, m in report.items():
        print(f"{k:>3} | {m['codec_pred_cos']:>9.3f} {m['codec_persist_cos']:>9.3f} "
              f"{m['codec_cos_gain']:>+9.3f}    | {m['codec_l1_skill']:>13.1%} | {m['latent_skill']:>11.1%}")
    print("\nPositive cos-gain / skill = forecasts the future better than persistence.")


if __name__ == "__main__":
    main()
