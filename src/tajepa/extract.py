"""Generic offline feature extraction & caching.

Any frontend exposing ``sample_rate``, ``frame_rate``, ``embedding_dim`` and
``encode([B,1,N]) -> [B,T,D]`` (the codec frontend and the log-mel frontend both
do) can be cached with this. Per-clip failures are isolated so a few corrupt files
(e.g. in FMA) don't abort a long run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import save_yaml
from .data.io import load_resampled
from .data.manifest import read_manifest


@torch.no_grad()
def extract_features(
    manifest_path: str | Path,
    cache_dir: str | Path,
    frontend: Any,
    name: str,
    extra_meta: dict | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries = read_manifest(manifest_path)
    iterator = entries
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(entries, desc=f"encode[{name}]")
        except ImportError:
            pass

    n_written = n_skipped = 0
    failures: list[dict] = []
    for entry in iterator:
        out_path = cache_dir / f"{entry.clip_id}.npy"
        if out_path.exists() and not overwrite:
            n_skipped += 1
            continue
        try:
            wav = load_resampled(entry.path, frontend.sample_rate)   # [1, N]
            emb = frontend.encode(wav.unsqueeze(0))                  # [1, T, D]
            np.save(out_path, emb.squeeze(0).cpu().numpy().astype(np.float32))
            n_written += 1
        except Exception as e:  # noqa: BLE001 - keep going past any decode/encode error
            failures.append({"clip_id": entry.clip_id, "path": entry.path, "error": repr(e)})

    if failures:
        import json

        with open(cache_dir / "failures.jsonl", "w") as f:
            for rec in failures:
                f.write(json.dumps(rec) + "\n")
        print(f"WARNING: {len(failures)} clip(s) failed; see {cache_dir / 'failures.jsonl'}")

    meta = {
        "frontend": name,
        "sample_rate": frontend.sample_rate,
        "frame_rate": frontend.frame_rate,
        "embedding_dim": frontend.embedding_dim,
        "n_clips": len(entries),
        "n_written": n_written,
        "n_skipped_existing": n_skipped,
        "n_failed": len(failures),
        "source_manifest": str(Path(manifest_path).resolve()),
        **(extra_meta or {}),
    }
    save_yaml(meta, cache_dir / "meta.yaml")
    return cache_dir
