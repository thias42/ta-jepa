"""Extract & cache frame-aligned MIR descriptors (Phase 2a control signals).

Mirrors extract_mel.py but for the descriptor frontend, so the controllable JEPA can
read cached descriptors paired with the cached codec embeddings.

    python scripts/extract_descriptors.py \
        --manifest data/manifests/esc50.jsonl --cache data/cache/descriptors/esc50
    # include pitch + voicing (slower, uses pyin):
    python scripts/extract_descriptors.py ... --names loudness centroid onset pitch voicing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.config import DescriptorConfig
from tajepa.features.descriptors import DescriptorFrontend
from tajepa.extract import extract_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--names", nargs="+", default=["loudness", "centroid", "onset"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = DescriptorConfig(names=tuple(args.names))
    frontend = DescriptorFrontend(cfg)
    out = extract_features(
        args.manifest, args.cache, frontend, name="descriptors",
        extra_meta={"names": list(cfg.names)}, overwrite=args.overwrite,
    )
    print(f"Cached descriptors {cfg.names} under {out}")


if __name__ == "__main__":
    main()
