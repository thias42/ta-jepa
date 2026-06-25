"""Extract & cache continuous codec embeddings for a manifest.

Example:
    python scripts/extract_embeddings.py \
        --manifest data/manifests/synthetic.jsonl \
        --cache data/cache/encodec_24khz/synthetic \
        --codec encodec_24khz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.config import CodecConfig
from tajepa.codec.extract import extract_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--codec", default="encodec_24khz")
    ap.add_argument("--hf-model-id", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = CodecConfig(name=args.codec, device=args.device)
    if args.hf_model_id:
        cfg.hf_model_id = args.hf_model_id

    out = extract_manifest(args.manifest, args.cache, cfg=cfg, overwrite=args.overwrite)
    print(f"Cached embeddings under {out} (see {out / 'meta.yaml'})")


if __name__ == "__main__":
    main()
