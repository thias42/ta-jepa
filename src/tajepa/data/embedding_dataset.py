"""Dataset over cached codec-embedding (or mel) sequences.

Phase 1 / the APC baseline train on sequences of continuous frame features. We
cache those features offline (see ``codec.extract``) as ``[T, D]`` ``.npy`` arrays,
then this dataset serves fixed-length windows. Keeping features cached on disk is
what lets the model side iterate fast (design note in the plan, Phase 0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import ManifestEntry, read_manifest


class EmbeddingSequenceDataset(Dataset):
    """Serves ``[T, D]`` windows from cached ``.npy`` feature files.

    ``cache_dir`` may be a single directory or a list of directories — the latter is how
    multi-domain pretraining is done (e.g. point at both the FMA and FSD50K caches at
    once). Feature dims must match across caches (same frontend).
    """

    def __init__(
        self,
        cache_dir: str | Path | list[str | Path],
        window_frames: int = 256,
        random_crop: bool = True,
        min_frames: int = 8,
        pattern: str = "*.npy",
    ) -> None:
        dirs = [cache_dir] if isinstance(cache_dir, (str, Path)) else list(cache_dir)
        self.cache_dirs = [Path(d) for d in dirs]
        self.files: list[Path] = []
        for d in self.cache_dirs:
            self.files.extend(sorted(d.rglob(pattern)))
        self.files.sort()
        if not self.files:
            roots = ", ".join(str(d) for d in self.cache_dirs)
            raise FileNotFoundError(f"No feature files matching {pattern} under: {roots}")
        self.window_frames = window_frames
        self.random_crop = random_crop
        self.min_frames = min_frames

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        arr = np.load(self.files[idx])              # [T, D]
        x = torch.from_numpy(arr).float()
        t = x.shape[0]
        w = self.window_frames
        if t > w:
            start = int(torch.randint(0, t - w + 1, (1,)).item()) if self.random_crop else 0
            x = x[start : start + w]
        return {"features": x, "length": x.shape[0], "clip_id": self.files[idx].stem}


class ManifestEmbeddingDataset(Dataset):
    """Cached features joined to manifest ``label`` / ``fold`` / ``split``.

    The bridge for held-out probe eval (e.g. ESC-50): it pairs each cached ``[T, D]``
    feature file with its manifest entry, optionally filtering by split, and exposes
    an integer-encoded label. Whole clips are returned (no random crop) since probes
    typically pool over time.
    """

    def __init__(
        self,
        manifest: str | Path | list[ManifestEntry],
        cache_dir: str | Path,
        split: str | None = None,
    ) -> None:
        entries = manifest if isinstance(manifest, list) else read_manifest(manifest)
        if split is not None:
            entries = [e for e in entries if e.split == split]
        self.cache_dir = Path(cache_dir)
        # Keep only entries whose features were actually cached.
        self.entries = [e for e in entries if (self.cache_dir / f"{e.clip_id}.npy").exists()]
        if not self.entries:
            raise FileNotFoundError(
                f"No cached features under {cache_dir} for manifest entries"
                + (f" with split={split}" if split else "")
            )
        labels = sorted({e.label for e in self.entries if e.label is not None})
        self.label_to_idx = {lab: i for i, lab in enumerate(labels)}

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        x = torch.from_numpy(np.load(self.cache_dir / f"{e.clip_id}.npy")).float()
        return {
            "features": x,
            "length": x.shape[0],
            "clip_id": e.clip_id,
            "label": e.label,
            "label_idx": self.label_to_idx.get(e.label, -1),
            "fold": e.fold,
        }


def pad_collate(batch: list[dict]) -> dict:
    """Collate variable-length ``[T, D]`` windows into a padded ``[B, T, D]`` batch
    plus a boolean ``pad_mask`` (True where padded)."""
    feats = [b["features"] for b in batch]
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    t_max = int(lengths.max())
    d = feats[0].shape[1]
    out = torch.zeros(len(feats), t_max, d)
    pad_mask = torch.ones(len(feats), t_max, dtype=torch.bool)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
        pad_mask[i, : f.shape[0]] = False
    return {
        "features": out,
        "lengths": lengths,
        "pad_mask": pad_mask,
        "clip_id": [b["clip_id"] for b in batch],
    }
