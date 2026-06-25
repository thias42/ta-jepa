"""Offline codec-embedding extraction & caching.

Runs the (frozen) codec frontend over every clip in a manifest and writes one
``[T, D]`` ``.npy`` per clip, plus a ``meta.yaml`` describing the cache (codec,
frame rate, dim). Caching offline is a deliberate Phase 0 choice: the codec pass
is the expensive part, and the model side then iterates over cheap cached arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import CodecConfig, save_yaml
from ..data.io import load_resampled
from ..data.manifest import read_manifest
from .frontend import CodecFrontend, build_frontend


@torch.no_grad()
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
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries = read_manifest(manifest_path)
    iterator = entries
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(entries, desc=f"encode[{cfg.name}]")
        except ImportError:
            pass

    n_written = 0
    n_skipped = 0
    failures: list[dict] = []
    for entry in iterator:
        out_path = cache_dir / f"{entry.clip_id}.npy"
        if out_path.exists() and not overwrite:
            n_skipped += 1
            continue
        # Resilient per-clip: real datasets (e.g. FMA) ship a handful of corrupt
        # files; one bad clip must not abort a multi-hour run.
        try:
            wav = load_resampled(entry.path, frontend.sample_rate)    # [1, N]
            emb = frontend.encode(wav.unsqueeze(0))                   # [1, T, D]
            np.save(out_path, emb.squeeze(0).cpu().numpy().astype(np.float32))
            n_written += 1
        except Exception as e:  # noqa: BLE001 - want to keep going past any decode/encode error
            failures.append({"clip_id": entry.clip_id, "path": entry.path, "error": repr(e)})

    if failures:
        import json

        with open(cache_dir / "failures.jsonl", "w") as f:
            for rec in failures:
                f.write(json.dumps(rec) + "\n")
        print(f"WARNING: {len(failures)} clip(s) failed; see {cache_dir / 'failures.jsonl'}")

    meta = {
        "codec": cfg.name,
        "hf_model_id": cfg.hf_model_id,
        "sample_rate": frontend.sample_rate,
        "frame_rate": frontend.frame_rate,
        "embedding_dim": frontend.embedding_dim,
        "n_clips": len(entries),
        "n_written": n_written,
        "n_skipped_existing": n_skipped,
        "n_failed": len(failures),
        "source_manifest": str(Path(manifest_path).resolve()),
    }
    save_yaml(meta, cache_dir / "meta.yaml")
    return cache_dir
