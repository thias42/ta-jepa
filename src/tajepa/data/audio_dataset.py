"""Dataset that yields fixed-length mono waveform chunks from a manifest.

Used by the offline embedding extractor and the mel baseline. It resamples to a
target rate and crops/pads to a fixed number of samples so batches are square.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .io import load_resampled
from .manifest import ManifestEntry, read_manifest


class AudioChunkDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | list[ManifestEntry],
        sample_rate: int = 24000,
        chunk_seconds: float = 4.0,
        random_crop: bool = True,
        mono: bool = True,
    ) -> None:
        self.entries = (
            manifest if isinstance(manifest, list) else read_manifest(manifest)
        )
        self.sample_rate = sample_rate
        self.chunk_len = int(round(chunk_seconds * sample_rate))
        self.random_crop = random_crop
        self.mono = mono

    def __len__(self) -> int:
        return len(self.entries)

    def _fit_length(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[-1]
        if n == self.chunk_len:
            return wav
        if n > self.chunk_len:
            start = (
                int(torch.randint(0, n - self.chunk_len + 1, (1,)).item())
                if self.random_crop
                else 0
            )
            return wav[..., start : start + self.chunk_len]
        return torch.nn.functional.pad(wav, (0, self.chunk_len - n))

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        wav = load_resampled(entry.path, self.sample_rate, mono=self.mono)  # [C, N]
        wav = self._fit_length(wav)
        return {"waveform": wav, "clip_id": entry.clip_id, "domain": entry.domain}
