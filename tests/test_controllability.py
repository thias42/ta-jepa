import torch

from tajepa.eval import controllability_matrix, disentanglement_report


class _FakeModel:
    """Pass-through model where the predicted/decoded frame equals the control delta,
    so perturbing control p must move measured descriptor p (a perfect diagonal)."""

    def __init__(self, cond_dim):
        self.cond_dim = cond_dim

    def eval(self):
        return self

    def to(self, _):
        return self

    def deltas_from(self, ctrl):
        return {1: ctrl}

    def predict_with_deltas(self, x, deltas, desc=None, pad_mask=None):
        return None, {1: deltas[1]}

    def reconstruct(self, p):
        return p


class _DS:
    def __init__(self, n, t, c):
        self.items = [
            {"features": torch.randn(t, 4), "control": torch.randn(t, c)} for _ in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_controllability_matrix_is_diagonal_for_passthrough():
    c = 3
    model = _FakeModel(c)
    ds = _DS(n=8, t=20, c=c)
    M, used = controllability_matrix(
        model, ds, render_fn=lambda s: s, desc_fn=lambda a: a,
        offset=1, bump=2.0, n_clips=8, device="cpu",
    )
    assert used == 8
    assert M.shape == (c, c)
    # diagonal ~ bump, off-diagonal ~ 0
    assert torch.allclose(torch.diagonal(M), torch.full((c,), 2.0), atol=1e-4)
    off = M - torch.diag(torch.diagonal(M))
    assert off.abs().max() < 1e-4


def test_disentanglement_report():
    M = torch.tensor([[2.0, 0.1, 0.0], [0.0, 1.5, 0.2], [0.1, 0.0, 1.8]])
    rep = disentanglement_report(M, names=["a", "b", "c"])
    assert rep["diag_positive"] == [True, True, True]
    assert rep["diagonal_dominant_frac"] == 1.0
    assert rep["dominance_ratio"] > 1.0
