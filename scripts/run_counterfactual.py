"""Counterfactual eval for an action-conditioned checkpoint (Phase 2).

Predicts each clip's future twice from the same context — once told the action that was
actually applied, once told nothing — and scores both against the *true intervened future*.
``action_gain = 1 - err(with) / err(without)``: positive means the model used the action,
~0 is a dead dial. Content difficulty cancels, since both predictions share a context and a
target.

    python scripts/run_counterfactual.py --ckpt runs/intervention.ckpt \
        --cache data/cache/interventions/fma_small --n-clips 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.embedding_dataset import InterventionPairDataset
from tajepa.eval import counterfactual_report
from tajepa.interventions import AXES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n-clips", type=int, default=300)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_intervention import InterventionLightning

    lit = InterventionLightning.load_from_checkpoint(str(args.ckpt), map_location="cpu")
    ds = InterventionPairDataset(args.cache, window_frames=args.window, random_crop=False)
    rep = counterfactual_report(lit.model, lit.target, ds, n_clips=args.n_clips,
                                device=args.device, axis_names=AXES)

    print(f"Counterfactual on {args.cache.name} ({len(ds)} pairs)")
    print("predict WITH the applied action vs WITHOUT; both scored against the true "
          "intervened future.\n")
    print(f"{'':<12}{'n':>6}{'err w/ action':>15}{'err w/o':>11}{'action_gain':>13}")
    for k, m in sorted(rep["overall"].items()):
        print(f"  k={k:<9}{m['n']:>6}{m['err_with_action']:>15.4f}"
              f"{m['err_without']:>11.4f}{m['action_gain']:>+12.1%}")

    if rep["per_axis"]:
        print("\nper-axis (clips where only that axis fired):")
        for ax, per in rep["per_axis"].items():
            n = next(iter(per.values()))["n"]
            row = "  ".join(f"k={k}: {v['action_gain']:+.1%}" for k, v in sorted(per.items()))
            print(f"  {ax:<12} n={n:<5} {row}")
    else:
        print("\n(no single-axis clips — per-axis attribution needs --p-axis lower)")


if __name__ == "__main__":
    main()
