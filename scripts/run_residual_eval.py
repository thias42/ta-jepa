"""Actions-controllability eval for a residual-action JEPA (Phase 2a+2b).

Keeps the real descriptor control fixed and varies only the learned action code, so the
effect signatures isolate what the *codes* do — the test of whether the residual codebook
captured non-loudness structure (transients/texture) instead of re-learning loudness.

    python scripts/run_residual_eval.py --ckpt runs/residual.ckpt \
        --features data/cache/encodec_24khz/esc50 --control data/cache/descriptors/esc50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import torch

from tajepa.config import CodecConfig, DescriptorConfig, resolve_device
from tajepa.codec.frontend import build_frontend
from tajepa.features.descriptors import DescriptorFrontend
from tajepa.data.embedding_dataset import PairedSequenceDataset
from tajepa.data.stats import ensure_codec_stats
from tajepa.eval import residual_action_effect_matrix, action_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--train-stats", type=Path, nargs="+", default=None,
                    help="Cache the model was TRAINED on. Only needed for pre-stats "
                         "checkpoints, whose grounding head records no output space; "
                         "without it the render is in the wrong space.")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--names", nargs="+", default=["loudness", "centroid", "onset"])
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or resolve_device("auto")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_residual import ResidualLightning, standardize

    model = ResidualLightning.load_from_checkpoint(str(args.ckpt), map_location="cpu").model
    ensure_codec_stats(model, args.train_stats, what=f"model {args.ckpt.name}")
    codec = build_frontend(CodecConfig(device=device))
    desc_fe = DescriptorFrontend(DescriptorConfig(names=tuple(args.names)))

    def render_fn(raw_codec):    # [B,T,Dc] raw codec -> audio [B,1,N]
        return codec.decode(raw_codec)

    def desc_fn(a):
        return desc_fe.encode(a.detach().cpu())

    def std_ctrl(c):
        return standardize(c, torch.zeros(c.shape[:2], dtype=torch.bool, device=c.device))

    ds = PairedSequenceDataset(args.features, args.control, window_frames=256, random_crop=False)
    res = residual_action_effect_matrix(model, ds, render_fn, desc_fn, std_ctrl,
                                        n_clips=args.n_clips, device=device)
    rep = action_report(res, names=args.names)
    M = res["effect"]
    print(f"Residual action-effect matrix ({M.shape[0]} codes, {args.n_clips} clips)")
    print("descriptor control held fixed; only the learned code is varied\n")
    hdr = "code | " + " | ".join(f"{n[:8]:>8}" for n in args.names) + " |  usage | consist | top"
    print(hdr); print("-" * len(hdr))
    for c in range(M.shape[0]):
        eff = " | ".join(f"{M[c, m].item():>+8.3f}" for m in range(M.shape[1]))
        print(f"{c:>4} | {eff} | {rep['usage'][c]:>6.2%} | {rep['consistency'][c]:>6.2f} | "
              f"{args.names[rep['dominant_descriptor'][c]]}")
    print(f"\nmean consistency: {rep['mean_consistency']:.2f}")
    print(f"separability: {rep['separability']:.2f}  (>1 = codes do distinct things)")
    tops = [args.names[rep["dominant_descriptor"][c]] for c in range(M.shape[0])]
    print(f"dominant-descriptor spread: " + ", ".join(f"{n}:{tops.count(n)}" for n in args.names))
    print("  (the win: codes whose top effect is NOT loudness = residual structure captured)")


if __name__ == "__main__":
    main()
