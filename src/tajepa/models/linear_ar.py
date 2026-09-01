"""Linear autoregression — the honest floor for forward prediction.

Persistence (``x[t+k] := x[t]``) is the weakest possible forecasting baseline: it uses
one frame and no fitting at all. Beating it demonstrates almost nothing, because codec
embeddings are locally smooth. A closed-form ridge regression on the last ``p`` frames
costs no training and is a far stronger reference — on ESC-50 transfer it reproduces the
causal JEPA's entire codec-space gain over persistence. Any claim that a model "learned
dynamics" has to clear *this* bar, not persistence.

Fitting is streaming (normal equations accumulated per clip), so it runs over a whole
cache without holding the design matrix in memory. Fit on the *training* distribution —
the same data the model saw — so a transfer evaluation is matched.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _contexts(x: torch.Tensor, order: int) -> torch.Tensor:
    """``[B,T,D]`` -> ``[B,T,order*D+1]``: the last ``order`` frames at each ``t``,
    flattened, plus a constant 1 for the bias. Positions before ``t=order-1`` reuse the
    first frame (edge padding), so a prediction is defined at every ``t``."""
    b, t, d = x.shape
    pad = x[:, :1].expand(b, order - 1, d)
    padded = torch.cat([pad, x], dim=1)                    # [B, T+order-1, D]
    win = padded.unfold(1, order, 1)                       # [B, T, D, order]
    flat = win.permute(0, 1, 3, 2).reshape(b, t, order * d)
    return torch.cat([flat, flat.new_ones(b, t, 1)], dim=-1)


class LinearAR(nn.Module):
    """Ridge AR(``order``) with one weight matrix per future offset.

    ``forward`` returns ``{offset: [B,T,D]}`` using the same alignment convention as the
    models it is compared against: entry ``t`` is the prediction of frame ``t+offset``.
    """

    def __init__(self, dim: int, order: int = 4, offsets: Iterable[int] = (1,),
                 ridge: float = 1e-2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.order = int(order)
        self.offsets = tuple(int(o) for o in offsets)
        self.ridge = float(ridge)
        n_feat = self.order * self.dim + 1
        for o in self.offsets:
            self.register_buffer(f"w_{o}", torch.zeros(n_feat, self.dim))
        self.register_buffer("fitted", torch.zeros(()))

    @torch.no_grad()
    def fit(self, sequences: Iterable[torch.Tensor]) -> "LinearAR":
        """Accumulate normal equations over ``[T,D]`` (or ``[B,T,D]``) sequences."""
        n_feat = self.order * self.dim + 1
        ata = {o: torch.zeros(n_feat, n_feat, dtype=torch.float64) for o in self.offsets}
        atb = {o: torch.zeros(n_feat, self.dim, dtype=torch.float64) for o in self.offsets}
        seen = 0
        for seq in sequences:
            x = seq if seq.dim() == 3 else seq.unsqueeze(0)
            x = x.detach().float().cpu()
            t = x.shape[1]
            ctx = _contexts(x, self.order)
            for o in self.offsets:
                if t <= o:
                    continue
                c = ctx[:, :-o].reshape(-1, n_feat).double()      # predicts t+o
                y = x[:, o:].reshape(-1, self.dim).double()
                ata[o] += c.T @ c
                atb[o] += c.T @ y
            seen += 1
        if seen == 0:
            raise ValueError("LinearAR.fit received no sequences")
        eye = torch.eye(n_feat, dtype=torch.float64)
        for o in self.offsets:
            w = torch.linalg.solve(ata[o] + self.ridge * ata[o].diagonal().mean() * eye, atb[o])
            getattr(self, f"w_{o}").copy_(w.float())
        self.fitted.fill_(1.0)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        if float(self.fitted) == 0.0:
            raise RuntimeError("LinearAR must be .fit() before use")
        ctx = _contexts(x, self.order)
        return {o: ctx @ getattr(self, f"w_{o}") for o in self.offsets}


    def save(self, path) -> None:
        """Persist the fitted reference so it can be shipped alongside a checkpoint.

        A fitted AR is a few hundred KB per offset, so the honest baseline can travel with
        the model instead of being re-fit at load time on whatever audio happens to be
        lying around — which is what makes a demo's reported skill reproducible.
        """
        torch.save({"dim": self.dim, "order": self.order, "offsets": list(self.offsets),
                    "ridge": self.ridge, "state_dict": self.state_dict()}, str(path))

    @classmethod
    def load(cls, path, map_location="cpu") -> "LinearAR":
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
        ar = cls(blob["dim"], order=blob["order"], offsets=blob["offsets"],
                 ridge=blob.get("ridge", 1e-2))
        ar.load_state_dict(blob["state_dict"])
        return ar


def _stride(n_total: int, n_want: int) -> range:
    """Evenly-spaced indices covering the whole dataset."""
    return range(0, n_total, max(1, n_total // max(1, n_want)))


@torch.no_grad()
def fit_linear_ar(
    dataset,
    dim: int,
    offsets: Iterable[int],
    order: int = 4,
    max_clips: int = 300,
    max_frames: int = 512,
    key: str = "features",
) -> LinearAR:
    """Fit a ``LinearAR`` on a feature dataset (dict items with ``key`` -> ``[T,D]``).

    Clips are drawn with an even **stride** across the dataset, never as the first
    ``max_clips``: a multi-cache dataset sorts by path, so its first N clips all come from
    one domain. Fitting the reference on a narrower distribution than the model saw would
    handicap the reference and flatter the model — the opposite of the point.
    """
    ar = LinearAR(dim, order=order, offsets=offsets)
    idx = _stride(len(dataset), max_clips)
    return ar.fit(dataset[i][key][:max_frames] for i in idx)
