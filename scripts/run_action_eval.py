"""Actions-controllability eval for an action-JEPA checkpoint (Phase 2b).

For each learned action code, forces it everywhere, renders the predicted latent back to
audio (EnCodec decoder), re-extracts the descriptors, and reports each code's effect
signature + consistency / distinctiveness / usage. See eval/action_controllability.py.

    python scripts/run_action_eval.py --ckpt runs/actions.ckpt \
        --features data/cache/encodec_24khz/esc50 --n-clips 40
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
from tajepa.data.embedding_dataset import EmbeddingSequenceDataset
from tajepa.eval import action_effect_matrix, action_report


def _global_codec_stats(cache_dir: Path, n: int = 200):
    files = sorted(Path(cache_dir).rglob("*.npy"))[:n]
    x = torch.from_numpy(np.concatenate([np.load(f) for f in files], axis=0)).float()
    return x.mean(0), x.std(0).clamp_min(1e-4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True, help="Codec cache dir.")
    ap.add_argument("--names", nargs="+", default=["loudness", "centroid", "onset"])
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or resolve_device("auto")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_actions import ActionLightning

    model = ActionLightning.load_from_checkpoint(str(args.ckpt), map_location="cpu").model
    codec = build_frontend(CodecConfig(device=device))
    mean, std = (t.to(device) for t in _global_codec_stats(args.features))
    desc_fe = DescriptorFrontend(DescriptorConfig(names=tuple(args.names)))

    def render_fn(std_codec):
        return codec.decode(std_codec * std + mean)

    def desc_fn(audio):
        return desc_fe.encode(audio.detach().cpu())

    ds = EmbeddingSequenceDataset(args.features, window_frames=256, random_crop=False)
    res = action_effect_matrix(model, ds, render_fn, desc_fn, n_clips=args.n_clips, device=device)
    rep = action_report(res, names=args.names)

    M = res["effect"]
    print(f"Action-effect matrix ({M.shape[0]} codes, {args.n_clips} clips)")
    print("each row = a code's effect on the re-extracted descriptors (vs inferred baseline)\n")
    hdr = "code | " + " | ".join(f"{n[:8]:>8}" for n in args.names) + " |  usage | consist | top"
    print(hdr); print("-" * len(hdr))
    for c in range(M.shape[0]):
        eff = " | ".join(f"{M[c, m].item():>+8.3f}" for m in range(M.shape[1]))
        print(f"{c:>4} | {eff} | {rep['usage'][c]:>6.2%} | {rep['consistency'][c]:>6.2f} | "
              f"{args.names[rep['dominant_descriptor'][c]]}")
    print(f"\nmean consistency: {rep['mean_consistency']:.2f}  (1.0 = noiseless, context-independent)")
    print(f"separability (between-code / within-code): {rep['separability']:.2f}  "
          f"(>1 = codes do distinct things)")
    used = sum(1 for u in rep["usage"] if u > 0.005)
    print(f"codes actually used by the inverse model: {used}/{M.shape[0]}")


if __name__ == "__main__":
    main()
