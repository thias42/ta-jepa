"""Lightweight YAML-backed config.

Configs are plain nested dicts loaded from YAML; small typed dataclasses give the
pieces that are referenced all over (codec frontend, mel frontend) a stable shape
and sane defaults. Anything experiment-specific stays as free-form dict keys so we
don't have to grow a dataclass every time a baseline gains a knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


@dataclass
class CodecConfig:
    """Configuration for the neural-codec embedding frontend.

    We take *continuous, pre-quantizer* encoder embeddings (design invariant #1),
    so ``bandwidth`` / number of quantizers is irrelevant here — we never touch the
    residual VQ. ``name`` selects the registered frontend implementation.
    """

    name: str = "encodec_24khz"            # registry key -> CodecFrontend impl
    hf_model_id: str = "facebook/encodec_24khz"
    sample_rate: int = 24000               # codec's native rate; audio is resampled to it
    # frame_rate (Hz) and embedding_dim are reported by the model after load; the
    # values below are the documented defaults for encodec_24khz and are only used
    # as a fallback / sanity check.
    expected_frame_rate: float = 75.0
    expected_embedding_dim: int = 128
    device: str = "auto"                   # auto -> cuda|mps|cpu


@dataclass
class MelConfig:
    """Log-mel frontend for the A-JEPA-comparable and APC-on-mel baselines."""

    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 320                  # ~75 Hz frames @ 24 kHz, matches encodec rate
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float | None = None
    log_offset: float = 1e-6
    normalize: bool = True                 # per-feature standardization at train time


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = [
    "load_yaml",
    "save_yaml",
    "CodecConfig",
    "MelConfig",
    "resolve_device",
    "asdict",
    "field",
]
