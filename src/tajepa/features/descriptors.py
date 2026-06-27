"""Frame-aligned MIR descriptors — the supervised control signals (Phase 2a).

Domain-general, interpretable axes defined across all audio (plan, Phase 2a): loudness,
spectral centroid (brightness), onset/transient density, and optionally pitch + a voicing
flag. Computed at the codec frame rate (75 Hz) so each descriptor frame lines up with a
codec-embedding frame, and cached like any other feature (``extract_features``). At train
time the *delta* of these descriptors conditions the predictor (see ``models/control``),
so control is learned as transition modulation, not absolute state.

Values are lightly transformed (log loudness/centroid/pitch) so they're ~O(1) and
roughly comparable; final standardization happens at train time.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import DescriptorConfig

# Order is fixed per name so the cached columns are stable. pitch/voicing are a pair.
_DIMS = {"loudness": 1, "centroid": 1, "onset": 1, "attack": 1, "pitch": 1, "voicing": 1}


class DescriptorFrontend:
    def __init__(self, cfg: DescriptorConfig | None = None) -> None:
        self.cfg = cfg or DescriptorConfig()
        bad = [n for n in self.cfg.names if n not in _DIMS]
        if bad:
            raise ValueError(f"unknown descriptor(s) {bad}; known: {sorted(_DIMS)}")
        self.sample_rate = self.cfg.sample_rate
        self.frame_rate = self.cfg.sample_rate / self.cfg.hop_length
        self.names = tuple(self.cfg.names)
        self.embedding_dim = sum(_DIMS[n] for n in self.names)

    def _one(self, y: np.ndarray) -> np.ndarray:
        import librosa

        c = self.cfg
        cols: dict[str, np.ndarray] = {}
        need_pitch = "pitch" in self.names or "voicing" in self.names
        if {"loudness", "centroid", "attack"} & set(self.names):
            S = np.abs(librosa.stft(y, n_fft=c.n_fft, hop_length=c.hop_length)) ** 2
        if {"loudness", "attack"} & set(self.names):
            rms = librosa.feature.rms(S=np.sqrt(S), frame_length=c.n_fft, hop_length=c.hop_length)[0]
            loud = np.log(rms + 1e-6)
        if "loudness" in self.names:
            cols["loudness"] = loud
        if "attack" in self.names:
            # smoothed positive rise-rate of log-energy: a stable transient/percussiveness
            # axis (cf. DAFx23 "attack time"), unlike spiky spectral-flux onset
            rise = np.maximum(np.diff(loud, prepend=loud[:1]), 0.0)
            cols["attack"] = np.convolve(rise, np.ones(3) / 3, mode="same")
        if "centroid" in self.names:
            cen = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=c.sample_rate)[0]
            cols["centroid"] = cen / (c.sample_rate / 2)            # normalized to Nyquist, in [0,1]
        if "onset" in self.names:
            cols["onset"] = librosa.onset.onset_strength(y=y, sr=c.sample_rate, hop_length=c.hop_length)
        if need_pitch:
            f0, voiced, _ = librosa.pyin(
                y, fmin=c.fmin_hz, fmax=c.fmax_hz, sr=c.sample_rate, hop_length=c.hop_length
            )
            if "pitch" in self.names:
                lp = np.log2(np.where(np.isfinite(f0) & (f0 > 0), f0, c.fmin_hz))
                cols["pitch"] = lp - np.log2(c.fmin_hz)              # octaves above fmin
            if "voicing" in self.names:
                cols["voicing"] = np.nan_to_num(voiced.astype(np.float32))

        t = min(len(cols[n]) for n in self.names)
        return np.stack([cols[n][:t] for n in self.names], axis=-1).astype(np.float32)  # [T, D]

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """``[B, 1, N]`` / ``[B, N]`` -> descriptors ``[B, T, D]`` (D = num descriptors)."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        outs = [self._one(waveform[b].cpu().numpy()) for b in range(waveform.shape[0])]
        t = min(o.shape[0] for o in outs)
        return torch.from_numpy(np.stack([o[:t] for o in outs], axis=0))  # [B, T, D]
