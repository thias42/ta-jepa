import copy

import numpy as np
import torch

from tajepa.models.control import ControllableJEPA, FiLM
from tajepa.models.jepa import jepa_loss
from tajepa.data.embedding_dataset import PairedSequenceDataset, pad_collate


def test_film_starts_identity_then_can_modulate():
    film = FiLM(cond_dim=3, feat_dim=8)
    h = torch.randn(2, 5, 8)
    c = torch.randn(2, 5, 3)
    # zero-init -> identity at start
    assert torch.allclose(film(h, c), h, atol=1e-6)
    # after a weight nudge it actually depends on c
    with torch.no_grad():
        film.to_gb.weight.add_(0.1 * torch.randn_like(film.to_gb.weight))
    assert not torch.allclose(film(h, c), film(h, torch.zeros_like(c)), atol=1e-5)


def test_controllable_forward_and_delta_shapes():
    b, t, d, cdim = 2, 32, 128, 3
    m = ControllableJEPA(in_dim=d, dim=64, enc_depth=2, pred_depth=2, heads=4,
                         offsets=(1, 3), cond_dim=cdim)
    x = torch.randn(b, t, d)
    desc = torch.randn(b, t, cdim)
    z, preds = m(x, desc)
    assert z.shape == (b, t, 64)
    assert set(preds) == {1, 3}
    for o in (1, 3):
        assert preds[o].shape == (b, t, 64)
    deltas = m.deltas_from(desc)
    assert deltas[1].shape == (b, t, cdim)


def test_control_delta_changes_prediction_after_training():
    # Train so FiLM is non-trivial, then check that changing the control delta changes
    # the predicted latent (i.e. the knob actually does something).
    torch.manual_seed(0)
    b, t, d, cdim = 6, 40, 32, 3
    m = ControllableJEPA(in_dim=d, dim=32, enc_depth=2, pred_depth=2, heads=4,
                         offsets=(2,), cond_dim=cdim)
    target = copy.deepcopy(m.encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    x = torch.randn(b, t, d)
    desc = torch.cumsum(0.2 * torch.randn(b, t, cdim), dim=1)  # smooth controls
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(60):
        z, preds = m(x, desc)
        with torch.no_grad():
            zt = target(x)
        loss, _ = jepa_loss(preds, z, zt, var_coef=1.0, cov_coef=0.04)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for c, tp in zip(m.encoder.parameters(), target.parameters()):
                tp.mul_(0.99).add_(c, alpha=0.01)

    with torch.no_grad():
        base = m.deltas_from(desc)
        bumped = {o: base[o].clone() for o in base}
        bumped[2][..., 0] += 2.0                       # perturb the first descriptor's delta
        _, p0 = m.predict_with_deltas(x, base)
        _, p1 = m.predict_with_deltas(x, bumped)
    assert (p0[2] - p1[2]).abs().mean() > 1e-3         # the knob steers the prediction


def test_paired_dataset_and_collate(tmp_path):
    feat, ctrl = tmp_path / "feat", tmp_path / "ctrl"
    for d in (feat, ctrl):
        d.mkdir()
    for cid in ("a", "b", "c"):
        np.save(feat / f"{cid}.npy", np.random.randn(300, 16).astype(np.float32))
        np.save(ctrl / f"{cid}.npy", np.random.randn(299, 3).astype(np.float32))  # off-by-one ok
    np.save(feat / "lonely.npy", np.random.randn(300, 16).astype(np.float32))     # no control -> dropped

    ds = PairedSequenceDataset(feat, ctrl, window_frames=128)
    assert len(ds) == 3
    item = ds[0]
    assert item["features"].shape == (128, 16)
    assert item["control"].shape == (128, 3)
    batch = pad_collate([ds[i] for i in range(3)])
    assert batch["features"].shape == (3, 128, 16)
    assert batch["control"].shape == (3, 128, 3)
