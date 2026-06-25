"""Run a linear probe on cached representations and report accuracy.

Codec-embedding baseline (the bar Phase 1 must beat), on ESC-50:
    python scripts/run_probe.py \
        --manifest data/manifests/esc50.jsonl \
        --cache data/cache/encodec_24khz/esc50 \
        --representation codec --pool meanstd

APC representation (needs a checkpoint from train_apc.py --save ...):
    python scripts/run_probe.py ... --representation apc --apc-ckpt runs/apc.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.embedding_dataset import ManifestEmbeddingDataset
from tajepa.eval import (
    IdentityRepresentation,
    APCRepresentation,
    AJEPARepresentation,
    run_linear_probe,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--representation", choices=["codec", "apc", "ajepa"], default="codec")
    ap.add_argument("--apc-ckpt", type=Path, default=None)
    ap.add_argument("--ajepa-ckpt", type=Path, default=None)
    ap.add_argument("--pool", choices=["mean", "meanstd"], default="meanstd")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    train_ds = ManifestEmbeddingDataset(args.manifest, args.cache, split=args.train_split)
    test_ds = ManifestEmbeddingDataset(args.manifest, args.cache, split=args.test_split)
    feat_dim = train_ds[0]["features"].shape[-1]

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if args.representation == "codec":
        rep = IdentityRepresentation(feat_dim)
        rep_name = "codec embeddings (identity)"
    elif args.representation == "apc":
        if args.apc_ckpt is None:
            ap.error("--representation apc requires --apc-ckpt")
        from train_apc import APCLightning

        lit = APCLightning.load_from_checkpoint(str(args.apc_ckpt), map_location="cpu")
        rep = APCRepresentation(lit.model)
        rep_name = f"APC hidden ({args.apc_ckpt.name})"
    else:
        if args.ajepa_ckpt is None:
            ap.error("--representation ajepa requires --ajepa-ckpt")
        from train_ajepa import AJEPALightning

        lit = AJEPALightning.load_from_checkpoint(str(args.ajepa_ckpt), map_location="cpu")
        rep = AJEPARepresentation(lit.ajepa)
        rep_name = f"A-JEPA patches ({args.ajepa_ckpt.name})"

    print(f"Probe: {rep_name} | pool={args.pool} | {args.train_split}->{args.test_split}")
    res = run_linear_probe(
        train_ds, test_ds, rep, pool=args.pool, epochs=args.epochs, lr=args.lr, device=args.device
    )
    print(f"  classes={res.num_classes}  feat_dim={res.feature_dim}  "
          f"n_train={res.n_train}  n_test={res.n_test}")
    print(f"  train acc = {res.train_acc:.4f}")
    print(f"  TEST  acc = {res.test_acc:.4f}   (chance = {1.0 / res.num_classes:.4f})")


if __name__ == "__main__":
    main()
