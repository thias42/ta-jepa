import torch

from tajepa.eval import action_effect_matrix, action_report


class _FakeActionModel:
    """Each code k adds a fixed, distinct vector to the decoded descriptors, so the
    effect matrix should be distinct per code and perfectly consistent across clips."""

    def __init__(self, num_codes, c):
        self.num_codes = num_codes
        # distinct per-code effect signatures baked into a [K, dim] table (dim>=c)
        self.table = torch.zeros(num_codes, 4)
        for k in range(num_codes):
            self.table[k, k % c] = float(k + 1)

    def eval(self):
        return self

    def to(self, _):
        return self

    def encode(self, x):
        return x

    def __call__(self, x):
        b, t, _ = x.shape
        return {"pred": torch.zeros(b, t, 4), "indices": torch.randint(0, self.num_codes, (b, t))}

    def predict_with_actions(self, x, codes):
        b, t = codes.shape
        eff = self.table[codes]                       # [b,t,4]
        return None, eff

    def reconstruct(self, p):
        return p


class _DS:
    def __init__(self, n, t):
        self.items = [{"features": torch.randn(t, 8)} for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_action_effect_matrix_distinct_and_consistent():
    c = 3
    model = _FakeActionModel(num_codes=4, c=c)
    ds = _DS(n=6, t=15)
    # render = identity (treat std_codec as audio); desc keeps the first c dims
    res = action_effect_matrix(
        model, ds, render_fn=lambda s: s, desc_fn=lambda a: a[..., :c], n_clips=6, device="cpu"
    )
    assert res["effect"].shape == (4, c)
    rep = action_report(res, names=["loud", "cen", "ons"])
    # deterministic effects -> perfectly consistent and clearly separable
    assert rep["mean_consistency"] > 0.99
    assert rep["separability"] > 5.0
    # code k's dominant descriptor is k % c (per the fake's table)
    assert rep["dominant_descriptor"] == [0, 1, 2, 0]
