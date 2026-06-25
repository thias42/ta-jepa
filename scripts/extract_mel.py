"""Extract & cache log-mel features for a manifest (A-JEPA baseline frontend).

Mirrors extract_embeddings.py but for the log-mel frontend, so A-JEPA trains on
cached mel sequences exactly the way APC trains on cached codec embeddings.

    python scripts/extract_mel.py \
        --manifest data/manifests/esc50.jsonl \
        --cache data/cache/logmel/esc50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.config import MelConfig, load_yaml
from tajepa.features.mel import LogMelFrontend
from tajepa.extract import extract_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None, help="mel YAML (configs/mel_baseline.yaml)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = MelConfig(**load_yaml(args.config)["mel"]) if args.config else MelConfig()
    frontend = LogMelFrontend(cfg)
    out = extract_features(
        args.manifest, args.cache, frontend, name="logmel",
        extra_meta={"n_mels": cfg.n_mels, "hop_length": cfg.hop_length},
        overwrite=args.overwrite,
    )
    print(f"Cached log-mel under {out} (see {out / 'meta.yaml'})")


if __name__ == "__main__":
    main()
