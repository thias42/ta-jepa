"""Neural-codec embedding frontend.

Design invariant #1: input features are *continuous, pre-quantizer* encoder
embeddings from a neural codec — never the discrete RVQ tokens, and never mel for
the main model. For EnCodec this means calling the encoder stack directly and
stopping before the residual vector quantizer.

The frontend is frozen (eval, no grad) and exposes a uniform interface so a DAC
implementation can drop in behind the same registry without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

from ..config import CodecConfig, resolve_device


class CodecFrontend(nn.Module, ABC):
    """Maps a mono waveform to a continuous ``[T, D]`` embedding sequence."""

    sample_rate: int
    frame_rate: float
    embedding_dim: int

    @abstractmethod
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """``waveform``: ``[B, 1, N]`` or ``[B, N]`` at ``self.sample_rate``.

        Returns continuous embeddings ``[B, T, D]`` (time-major per item).
        """

    @torch.no_grad()
    def encode_numpy(self, waveform: torch.Tensor) -> "torch.Tensor":
        return self.encode(waveform)


_REGISTRY: dict[str, Callable[[CodecConfig], CodecFrontend]] = {}


def register_frontend(name: str):
    def deco(fn: Callable[[CodecConfig], CodecFrontend]):
        _REGISTRY[name] = fn
        return fn

    return deco


def build_frontend(cfg: CodecConfig) -> CodecFrontend:
    if cfg.name not in _REGISTRY:
        raise KeyError(f"Unknown codec frontend '{cfg.name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.name](cfg)


class EncodecFrontend(CodecFrontend):
    """EnCodec continuous pre-quantizer embeddings via HF ``transformers``.

    ``model.encoder(x)`` returns ``[B, D, T]`` *before* the residual VQ — exactly
    the continuous latent we want. We never call ``.encode()`` (which quantizes).
    """

    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        from transformers import EncodecModel

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        model = EncodecModel.from_pretrained(cfg.hf_model_id)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(self.device)

        self.sample_rate = int(getattr(model.config, "sampling_rate", cfg.sample_rate))
        self.frame_rate = float(getattr(model.config, "frame_rate", cfg.expected_frame_rate))
        self.embedding_dim = int(getattr(model.config, "hidden_size", cfg.expected_embedding_dim))
        # 24 kHz model does not normalize; 48 kHz does. Honor whatever the config says.
        self.normalize = bool(getattr(model.config, "normalize", False))

    def _prep(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:                  # [B, N] -> [B, 1, N]
            waveform = waveform.unsqueeze(1)
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(f"Expected mono [B,1,N] or [B,N], got {tuple(waveform.shape)}")
        return waveform.to(self.device)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self._prep(waveform)
        if self.normalize:
            mono = x.mean(1, keepdim=True)
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            x = x / scale
        emb = self.model.encoder(x)              # [B, D, T]
        return emb.transpose(1, 2).contiguous()  # [B, T, D]

    @torch.no_grad()
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Render continuous (pre-quantizer) embeddings ``[B, T, D]`` back to audio
        ``[B, 1, N]`` via the codec's own decoder — skipping the quantizer (the optional
        render stage in the plan). Used by the closed-loop controllability eval."""
        emb = embeddings.to(self.device).transpose(1, 2)   # [B, D, T]
        return self.model.decoder(emb)


@register_frontend("encodec_24khz")
@register_frontend("encodec_48khz")
@register_frontend("encodec")
def _build_encodec(cfg: CodecConfig) -> CodecFrontend:
    return EncodecFrontend(cfg)
