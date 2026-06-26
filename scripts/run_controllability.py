"""Closed-loop controllability eval for a controllable-JEPA checkpoint (Phase 2a).

Perturbs each control descriptor's delta, renders the prediction back to audio (EnCodec
decoder), re-extracts the descriptors, and reports the controllability matrix +
disentanglement summary. See src/tajepa/eval/controllability.py.

    python scripts/run_controllability.py --ckpt runs/control.ckpt \
        --features data/cache/encodec_24khz/esc50 --control data/cache/descriptors/esc50 \
        --offset 1 --bump 2.0 --n-clips 50
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
from tajepa.eval import controllability_matrix, disentanglement_report


def _global_codec_stats(cache_dir: Path, n: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    files = sorted(Path(cache_dir).rglob("*.npy"))[:n]
    arr = np.concatenate([np.load(f) for f in files], axis=0)
    x = torch.from_numpy(arr).float()
    return x.mean(0), x.std(0).clamp_min(1e-4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True, help="Codec cache dir.")
    ap.add_argument("--control", type=Path, required=True, help="Descriptor cache dir.")
    ap.add_argument("--names", nargs="+", default=["loudness", "centroid", "onset"])
    ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--bump", type=float, default=2.0)
    ap.add_argument("--n-clips", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or resolve_device("auto")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_control import ControlLightning, standardize

    lit = ControlLightning.load_from_checkpoint(str(args.ckpt), map_location="cpu")
    model = lit.model

    codec = build_frontend(CodecConfig(device=device))
    mean, std = _global_codec_stats(args.features)
    mean, std = mean.to(device), std.to(device)
    desc_fe = DescriptorFrontend(DescriptorConfig(names=tuple(args.names)))

    def render_fn(std_codec: torch.Tensor) -> torch.Tensor:    # [B,T,Dc] -> audio [B,1,N]
        return codec.decode(std_codec * std + mean)

    def desc_fn(audio: torch.Tensor) -> torch.Tensor:           # audio -> [B,T,C] (on cpu)
        return desc_fe.encode(audio.detach().cpu())

    ds = PairedSequenceDataset(args.features, args.control, window_frames=256, random_crop=False)

    def std_ctrl(c):
        return standardize(c, torch.zeros(c.shape[:2], dtype=torch.bool, device=c.device))

    M, used = controllability_matrix(
        model, ds, render_fn, desc_fn, offset=args.offset, bump=args.bump,
        n_clips=args.n_clips, device=device, standardize_fn=std_ctrl,
    )
    # Normalize each measured column by its own scale so off-diagonals are comparable.
    M = M / M.abs().max(dim=0).values.clamp_min(1e-8)
    rep = disentanglement_report(M, names=args.names)

    print(f"Controllability matrix (offset={args.offset}, bump=+{args.bump}, {used} clips)")
    print("rows = perturbed control, cols = measured descriptor (col-normalized)\n")
    hdr = "perturb \\ measure | " + " | ".join(f"{n[:8]:>8}" for n in args.names)
    print(hdr); print("-" * len(hdr))
    for p, name in enumerate(args.names):
        print(f"{name:>17} | " + " | ".join(f"{M[p, m].item():>+8.3f}" for m in range(len(args.names))))
    print(f"\ndiagonal positive: {rep['diag_positive']}")
    print(f"diagonal-dominant fraction: {rep['diagonal_dominant_frac']:.2f}  "
          f"(1.0 = each control most affects its own descriptor)")
    print(f"dominance ratio (|diag|/|off|): {rep['dominance_ratio']:.2f}")


if __name__ == "__main__":
    main()
