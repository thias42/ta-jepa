"""Audio loading.

We read with ``soundfile`` rather than ``torchaudio.load`` on purpose: recent
torchaudio delegates decoding to TorchCodec/FFmpeg, an extra native dependency we
don't want in the Phase 0 path. ``soundfile`` (libsndfile) handles wav/flac/ogg.
Resampling uses ``torchaudio.transforms.Resample``, which is a pure tensor op with
no codec backend.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import torch


def load_audio(path: str | Path, mono: bool = True) -> tuple[torch.Tensor, int]:
    """Return ``(waveform[C, N] float32, sample_rate)``. Mono-mixes if requested."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [N, C]
    wav = torch.from_numpy(data).T.contiguous()                     # [C, N]
    if mono and wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav, int(sr)


@lru_cache(maxsize=32)
def _resampler(orig_sr: int, target_sr: int):
    import torchaudio

    return torchaudio.transforms.Resample(orig_sr, target_sr)


def load_resampled(path: str | Path, target_sr: int, mono: bool = True) -> torch.Tensor:
    wav, sr = load_audio(path, mono=mono)
    if sr != target_sr:
        wav = _resampler(sr, target_sr)(wav)
    return wav
