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
_DIMS = {"loudness": 1, "centroid": 1, "onset": 1, "attack": 1, "attack_time": 1,
         "harmonic_ratio": 1, "pitch": 1, "voicing": 1}


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
        if {"loudness", "centroid", "attack", "attack_time", "harmonic_ratio", "pitch"} & set(self.names):
            S = np.abs(librosa.stft(y, n_fft=c.n_fft, hop_length=c.hop_length)) ** 2
        if {"loudness", "attack", "attack_time"} & set(self.names):
            rms = librosa.feature.rms(S=np.sqrt(S), frame_length=c.n_fft, hop_length=c.hop_length)[0]
            loud = np.log(rms + 1e-6)
            rise = np.maximum(np.diff(loud, prepend=loud[:1]), 0.0)
        if "loudness" in self.names:
            cols["loudness"] = loud
        if "attack" in self.names:
            # smoothed positive rise-RATE of log-energy (entangled with loudness magnitude)
            cols["attack"] = np.convolve(rise, np.ones(3) / 3, mode="same")
        if "attack_time" in self.names:
            # loudness-NORMALIZED rise sharpness (cf. DAFx23 attack time): normalize the rise
            # by the local log-energy range so it measures "how sharp", decorrelated from "how
            # loud" (|corr with loudness| ~0.22 vs ~0.84 for spectral flatness).
            from scipy.ndimage import maximum_filter1d, minimum_filter1d
            win = 15
            rng = (maximum_filter1d(loud, win) - minimum_filter1d(loud, win)) + 1e-3
            cols["attack_time"] = np.convolve(rise / rng, np.ones(3) / 3, mode="same")
        if "centroid" in self.names:
            cen = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=c.sample_rate)[0]
            cols["centroid"] = cen / (c.sample_rate / 2)            # normalized to Nyquist, in [0,1]
        if "onset" in self.names:
            cols["onset"] = librosa.onset.onset_strength(y=y, sr=c.sample_rate, hop_length=c.hop_length)
        # harmonic ratio (HPSS): spectral/timbral "tonal vs noisy" axis — codec-recoverable
        # and render-friendly, unlike transients. Also gates pitch (meaningful when tonal).
        hr = None
        if {"harmonic_ratio", "pitch"} & set(self.names):
            mag = np.sqrt(S)
            Hc, Pc = librosa.decompose.hpss(mag)
            hr = Hc.sum(0) / (Hc.sum(0) + Pc.sum(0) + 1e-9)
        if "harmonic_ratio" in self.names:
            cols["harmonic_ratio"] = hr
        if "pitch" in self.names:
            # fast f0 via yin (pyin is too slow at scale); octaves above fmin, gated to 0
            # where the frame isn't tonal (harmonic ratio low) so pitch is meaningful.
            f0 = librosa.yin(y, fmin=c.fmin_hz, fmax=c.fmax_hz, sr=c.sample_rate,
                             hop_length=c.hop_length)
            p = np.log2(np.clip(f0, c.fmin_hz, c.fmax_hz) / c.fmin_hz)
            L = min(len(p), len(hr))
            cols["pitch"] = np.where(hr[:L] > 0.5, p[:L], 0.0)
        if "voicing" in self.names:
            _, voiced, _ = librosa.pyin(
                y, fmin=c.fmin_hz, fmax=c.fmax_hz, sr=c.sample_rate, hop_length=c.hop_length
            )
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
