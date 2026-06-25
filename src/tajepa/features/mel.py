"""Log-mel frontend.

Two uses: (1) the A-JEPA-comparable mel baseline for X-ARES, and (2) APC-on-mel,
the original APC setting, as a sanity reference for our APC reimplementation. Hop
length defaults to 320 @ 24 kHz so mel frames land at ~75 Hz, matching the EnCodec
frame rate and keeping the two frontends interchangeable downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import MelConfig


class LogMelFrontend(nn.Module):
    def __init__(self, cfg: MelConfig | None = None) -> None:
        super().__init__()
        import torchaudio

        self.cfg = cfg or MelConfig()
        self.embedding_dim = self.cfg.n_mels
        self.sample_rate = self.cfg.sample_rate
        self.frame_rate = self.cfg.sample_rate / self.cfg.hop_length
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            f_min=self.cfg.f_min,
            f_max=self.cfg.f_max,
            power=2.0,
        )

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """``[B, 1, N]`` or ``[B, N]`` -> log-mel ``[B, T, n_mels]``."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        mel = self.melspec(waveform)                       # [B, n_mels, T]
        logmel = torch.log(mel + self.cfg.log_offset)
        return logmel.transpose(1, 2).contiguous()         # [B, T, n_mels]
