"""Offline codec-embedding extraction & caching.

Thin wrapper over :func:`tajepa.extract.extract_features` that builds the (frozen)
codec frontend. Caching offline is a deliberate Phase 0 choice: the codec pass is
the expensive part, and the model side then iterates over cheap cached arrays.
"""

from __future__ import annotations

from pathlib import Path

from ..config import CodecConfig
from ..extract import extract_features
from .frontend import CodecFrontend, build_frontend


def extract_manifest(
    manifest_path: str | Path,
    cache_dir: str | Path,
    cfg: CodecConfig | None = None,
    frontend: CodecFrontend | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    cfg = cfg or CodecConfig()
    frontend = frontend or build_frontend(cfg)
    return extract_features(
        manifest_path,
        cache_dir,
        frontend,
        name=cfg.name,
        extra_meta={"hf_model_id": cfg.hf_model_id},
        overwrite=overwrite,
        progress=progress,
    )
