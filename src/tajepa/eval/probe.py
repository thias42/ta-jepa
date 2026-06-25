"""X-ARES-style linear probe.

Protocol: freeze the representation, mean(-std)-pool over time to one vector per
clip, fit a linear classifier on the train split, report accuracy on the held-out
split. Used in Phase 0 to get a real ESC-50 number from the raw codec embeddings
(the baseline Phase 1's JEPA encoder must beat) and, later, to score the JEPA
encoder itself through the very same code path.

Standardization (z-score with train statistics) is applied before the linear layer,
as is standard for linear probes — it makes the linear fit well-conditioned without
adding any capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..config import resolve_device
from .representations import Representation


def pool_time(seq: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """``[B, T, R]`` -> ``[B, R]`` (mean) or ``[B, 2R]`` (meanstd)."""
    if mode == "mean":
        return seq.mean(dim=1)
    if mode == "meanstd":
        return torch.cat([seq.mean(dim=1), seq.std(dim=1)], dim=-1)
    raise ValueError(f"unknown pool mode {mode!r}")


@torch.no_grad()
def extract_pooled(
    dataset: Dataset,
    representation: Representation,
    pool: str = "mean",
    device: str | None = None,
) -> tuple[torch.Tensor, list[str], list[int | None]]:
    """Pool every clip to one vector. Returns ``(X [N, R'], labels, folds)``."""
    device = device or resolve_device("auto")
    if isinstance(representation, nn.Module) or hasattr(representation, "model"):
        getattr(representation, "model", representation).to(device)
    feats_out, labels, folds = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        x = item["features"].unsqueeze(0).to(device)   # [1, T, D]
        rep = representation(x)                         # [1, T, R]
        feats_out.append(pool_time(rep, pool).squeeze(0).cpu())
        labels.append(item["label"])
        folds.append(item.get("fold"))
    return torch.stack(feats_out), labels, folds


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, weight_decay: float = 1e-4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        self.weight_decay = weight_decay
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.standardize(x))

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 300,
        lr: float = 1e-2,
        device: str | None = None,
        verbose: bool = False,
    ) -> "LinearProbe":
        device = device or resolve_device("auto")
        self.to(device)
        x, y = x.to(device), y.to(device)
        self.mu = x.mean(0)
        self.sd = x.std(0).clamp_min(1e-6)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self(x), y)
            loss.backward()
            opt.step()
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:4d}  loss {loss.item():.4f}")
        return self

    @torch.no_grad()
    def score(self, x: torch.Tensor, y: torch.Tensor, device: str | None = None) -> float:
        device = device or resolve_device("auto")
        self.eval()
        pred = self(x.to(device)).argmax(-1).cpu()
        return float((pred == y).float().mean())


@dataclass
class ProbeResult:
    train_acc: float
    test_acc: float
    num_classes: int
    feature_dim: int
    n_train: int
    n_test: int


def run_linear_probe(
    train_ds: Dataset,
    test_ds: Dataset,
    representation: Representation,
    pool: str = "mean",
    epochs: int = 300,
    lr: float = 1e-2,
    device: str | None = None,
    verbose: bool = False,
) -> ProbeResult:
    device = device or resolve_device("auto")
    xtr, ltr, _ = extract_pooled(train_ds, representation, pool, device)
    xte, lte, _ = extract_pooled(test_ds, representation, pool, device)

    # One label->index map shared across splits, derived from the train labels.
    classes = sorted({lab for lab in ltr if lab is not None})
    idx = {c: i for i, c in enumerate(classes)}
    ytr = torch.tensor([idx[l] for l in ltr], dtype=torch.long)
    yte = torch.tensor([idx.get(l, -1) for l in lte], dtype=torch.long)

    probe = LinearProbe(xtr.shape[1], len(classes))
    probe.fit(xtr, ytr, epochs=epochs, lr=lr, device=device, verbose=verbose)
    return ProbeResult(
        train_acc=probe.score(xtr, ytr, device),
        test_acc=probe.score(xte, yte, device),
        num_classes=len(classes),
        feature_dim=xtr.shape[1],
        n_train=len(ytr),
        n_test=len(yte),
    )
