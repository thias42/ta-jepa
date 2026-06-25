"""Representation providers for the linear probe.

A ``Representation`` maps a frame-feature sequence ``[B, T, D]`` to a (frozen)
representation sequence ``[B, T, R]`` to be pooled and probed. This indirection is
what lets the *same* probe score the raw codec embeddings today and the APC / Phase-1
JEPA encoders later — they all just present this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Representation(Protocol):
    dim: int

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:  # [B,T,D] -> [B,T,R]
        ...


class IdentityRepresentation:
    """Probe the input features directly — i.e. the codec-embedding baseline.

    This is the bar Phase 1 must beat: how linearly separable are the raw EnCodec
    embeddings before any predictive pretraining?
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        return feats


class APCRepresentation:
    """Frozen APC hidden states as the representation (the standard APC probe)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(model.hidden_dim)

    @torch.no_grad()
    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        _, repr_seq = self.model(feats)
        return repr_seq


class AJEPARepresentation:
    """Frozen A-JEPA context-encoder patch tokens (the A-JEPA probe).

    Returns the full (unmasked) patch representations ``[B, N, dim]``; the probe
    pools over the ``N`` patch axis just as it pools over time for the others.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(model.dim)

    @torch.no_grad()
    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model.encode_full(feats)


class JEPARepresentation:
    """Frozen causal JEPA frame encoder ``f_θ`` (the Phase 1 probe).

    Returns the online latent sequence ``[B, T, dim]``; the probe pools over time.
    This is what the Phase 1 gate measures against the APC / codec baselines.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(model.dim)

    @torch.no_grad()
    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model.encode(feats)
