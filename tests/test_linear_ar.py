import numpy as np
import torch

from tajepa.data.stats import codec_stats, ensure_codec_stats
from tajepa.models.jepa import JEPA
from tajepa.models.linear_ar import LinearAR, _contexts, fit_linear_ar


def test_context_window_alignment():
    # ctx[:, t] must hold frames t-order+1..t (most recent last), then a bias 1.
    x = torch.randn(2, 12, 3)
    c = _contexts(x, order=4)
    assert c.shape == (2, 12, 4 * 3 + 1)
    assert torch.allclose(c[:, 7, 9:12], x[:, 7])
    assert torch.allclose(c[:, 7, 6:9], x[:, 6])
    assert torch.allclose(c[:, :, -1], torch.ones(2, 12))
    # early positions edge-pad with the first frame rather than leaking the future
    assert torch.allclose(c[:, 0, 0:3], x[:, 0])


def test_recovers_a_known_ar_process():
    torch.manual_seed(0)
    w = torch.randn(5, 5) * 0.3
    seqs = []
    for _ in range(20):
        z = torch.randn(1, 200, 5)
        for t in range(1, 200):
            z[0, t] = z[0, t - 1] @ w
        seqs.append(z)
    ar = LinearAR(5, order=4, offsets=(1, 2)).fit(seqs)
    pred = ar(seqs[0])
    assert (pred[1][:, :-1] - seqs[0][:, 1:]).abs().mean() < 1e-3
    assert (pred[2][:, :-2] - seqs[0][:, 2:]).abs().mean() < 1e-3


def test_beats_persistence_on_an_ar_signal():
    torch.manual_seed(0)
    d = 6
    w = torch.randn(d, d) * 0.25
    seqs = []
    for _ in range(15):
        z = torch.randn(1, 150, d)
        for t in range(1, 150):
            z[0, t] = z[0, t - 1] @ w + 0.1 * z[0, t]
        seqs.append(z)
    ar = LinearAR(d, order=4, offsets=(1,)).fit(seqs)
    p = ar(seqs[0])[1][:, :-1]
    ar_err = (p - seqs[0][:, 1:]).abs().mean()
    persist_err = (seqs[0][:, :-1] - seqs[0][:, 1:]).abs().mean()
    assert ar_err < persist_err


def test_must_be_fitted_before_use():
    ar = LinearAR(4, order=2, offsets=(1,))
    try:
        ar(torch.randn(1, 10, 4))
    except RuntimeError as e:
        assert "fit" in str(e)
    else:
        raise AssertionError("unfitted LinearAR should refuse to predict")


def test_fit_linear_ar_from_a_dataset():
    class DS:
        def __init__(self):
            self.q = [torch.cumsum(0.1 * torch.randn(40, 4), 0) for _ in range(6)]

        def __len__(self):
            return len(self.q)

        def __getitem__(self, i):
            return {"features": self.q[i]}

    ar = fit_linear_ar(DS(), dim=4, offsets=(1, 3), order=3)
    assert set(ar(torch.randn(1, 20, 4))) == {1, 3}


def test_codec_stats_and_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(4):
        np.save(tmp_path / f"c{i}.npy", (rng.normal(3.0, 2.0, (50, 8))).astype("float32"))
    mean, std = codec_stats(tmp_path)
    assert mean.shape == (8,) and std.shape == (8,)
    assert abs(float(mean.mean()) - 3.0) < 0.3
    assert abs(float(std.mean()) - 2.0) < 0.3

    m = JEPA(in_dim=8, dim=16, enc_depth=1, pred_depth=1, heads=2, offsets=(1,))
    assert not m.has_codec_stats
    m.set_codec_stats(mean, std)
    assert m.has_codec_stats
    # reconstruct_raw un-does exactly the standardization the grounding target applied
    z = torch.randn(2, 5, 16)
    assert torch.allclose(m.reconstruct_raw(z), m.reconstruct(z) * std + mean, atol=1e-5)
    x = torch.randn(2, 5, 8) * std + mean
    assert torch.allclose(m.standardize_codec(x) * std + mean, x, atol=1e-4)
    # statistics survive a state_dict round trip (i.e. they travel in the checkpoint)
    m2 = JEPA(in_dim=8, dim=16, enc_depth=1, pred_depth=1, heads=2, offsets=(1,))
    m2.load_state_dict(m.state_dict())
    assert torch.allclose(m2.codec_mean, mean) and m2.has_codec_stats


def test_ensure_codec_stats_warns_without_a_source(recwarn):
    m = JEPA(in_dim=8, dim=16, enc_depth=1, pred_depth=1, heads=2, offsets=(1,))
    assert ensure_codec_stats(m, None) is False
    assert any("no codec statistics" in str(w.message) for w in recwarn)


def test_ar_fit_strides_across_a_multi_cache_dataset(tmp_path):
    """A pooled multi-cache dataset sorts by path, so its first N clips all come from one
    domain. Fitting the reference on a narrower distribution than the model saw would
    handicap the reference and flatter the model, so sampling must stride."""
    import collections

    from tajepa.data.embedding_dataset import EmbeddingSequenceDataset
    from tajepa.models.linear_ar import _stride

    rng = np.random.default_rng(0)
    a, b = tmp_path / "aaa_big", tmp_path / "zzz_small"
    for d, n in ((a, 80), (b, 20)):
        d.mkdir()
        for i in range(n):
            np.save(d / f"{i:03d}.npy", rng.normal(size=(30, 4)).astype("float32"))

    ds = EmbeddingSequenceDataset([a, b], window_frames=30, random_crop=False)
    idx = list(_stride(len(ds), 20))
    seen = collections.Counter("big" if "aaa_big" in str(ds.files[i]) else "small" for i in idx)
    assert seen["small"] > 0, "strided sample must reach the cache that sorts last"
    # and roughly in proportion to cache size (80:20), not 50/50 or 100/0
    assert 2.0 < seen["big"] / seen["small"] < 6.0

    # taking the first N instead would see only one domain — the bug this guards
    first = collections.Counter(
        "big" if "aaa_big" in str(ds.files[i]) else "small" for i in range(20))
    assert first["small"] == 0


def test_saved_ar_round_trips(tmp_path):
    torch.manual_seed(0)
    seqs = [torch.randn(1, 60, 5) for _ in range(6)]
    ar = LinearAR(5, order=3, offsets=(1, 4)).fit(seqs)
    path = tmp_path / "ar.pt"
    ar.save(path)
    back = LinearAR.load(path)
    assert back.order == 3 and back.dim == 5 and back.offsets == (1, 4)
    x = torch.randn(1, 40, 5)
    for k in (1, 4):
        assert torch.allclose(ar(x)[k], back(x)[k], atol=1e-6)
