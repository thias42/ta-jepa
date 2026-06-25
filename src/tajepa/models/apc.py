"""APC baseline — Autoregressive Predictive Coding.

The known-good causal-prediction reference for Phase 0 (plan, Phase 0): a
unidirectional multi-layer LSTM with inter-layer residual connections that, from
``x[:t]``, predicts a future frame ``x[t+n]`` and is trained with an L1 loss on the
*actual* frame. We also support predicting several offsets jointly (Multi-Target
APC), which the plan calls for to discourage trivial local-smoothness solutions.

Crucially this is NOT our anti-collapse template (CLAUDE.md): APC cannot collapse
because it regresses a grounded input frame. Phase 1's JEPA predicts a moving EMA
target and *can* collapse — different problem, handled there. Here we keep APC
faithful to the original so it is a fair, well-understood baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class APCModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        offsets: tuple[int, ...] = (3,),
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if not offsets or any(n < 1 for n in offsets):
            raise ValueError(f"offsets must be positive ints, got {offsets}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.offsets = tuple(offsets)
        self.residual = residual and (hidden_dim == hidden_dim)  # residual needs matching dims

        self.input_proj = (
            nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        )
        self.rnns = nn.ModuleList(
            [nn.LSTM(hidden_dim, hidden_dim, batch_first=True) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        # One prediction head per future offset (Multi-Target APC).
        self.heads = nn.ModuleDict(
            {str(n): nn.Linear(hidden_dim, input_dim) for n in self.offsets}
        )

    def forward(self, x: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """``x``: ``[B, T, input_dim]``.

        Returns ``(preds, repr)`` where ``preds[n]`` is ``[B, T, input_dim]`` (the
        prediction of ``x[t+n]`` made at each ``t``) and ``repr`` is the final
        hidden sequence ``[B, T, hidden_dim]`` used for downstream linear probing.
        """
        h = self.input_proj(x)
        for rnn in self.rnns:
            out, _ = rnn(h)
            out = self.dropout(out)
            h = h + out if (self.residual and out.shape[-1] == h.shape[-1]) else out
        preds = {n: self.heads[str(n)](h) for n in self.offsets}
        return preds, h


def apc_loss(
    preds: dict[int, torch.Tensor],
    x: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean L1 over offsets between ``preds[n][:, :-n]`` and ``x[:, n:]``.

    ``pad_mask``: ``[B, T]`` True where padded; excluded from the loss.
    Returns ``(loss, logs)``.
    """
    total = x.new_zeros(())
    logs: dict[str, float] = {}
    for n, pred in preds.items():
        if x.shape[1] <= n:
            continue
        p = pred[:, :-n]                 # prediction of frame t+n, aligned at t
        tgt = x[:, n:]                   # actual future frame
        l1 = (p - tgt).abs()             # [B, T-n, D]
        if pad_mask is not None:
            valid = (~pad_mask[:, n:]).unsqueeze(-1)   # [B, T-n, 1]
            denom = valid.sum().clamp(min=1) * x.shape[-1]
            loss_n = (l1 * valid).sum() / denom
        else:
            loss_n = l1.mean()
        total = total + loss_n
        logs[f"l1_n{n}"] = float(loss_n.detach())
    loss = total / max(1, len(preds))
    logs["loss"] = float(loss.detach())
    return loss, logs


@torch.no_grad()
def persistence_l1(
    x: torch.Tensor, offset: int, pad_mask: torch.Tensor | None = None
) -> float:
    """Naive persistence baseline: predict ``x[t+n] := x[t]``. The bar Phase 1 must
    clear (plan: "beats persistence"). Useful as a reference alongside APC too."""
    if x.shape[1] <= offset:
        return float("nan")
    l1 = (x[:, offset:] - x[:, :-offset]).abs()
    if pad_mask is not None:
        valid = (~pad_mask[:, offset:]).unsqueeze(-1)
        return float((l1 * valid).sum() / (valid.sum().clamp(min=1) * x.shape[-1]))
    return float(l1.mean())
