import torch

from tajepa.models.apc import APCModel, apc_loss, persistence_l1
from tajepa.diagnostics import collapse_report


def test_apc_shapes_and_offsets():
    b, t, d = 4, 32, 16
    x = torch.randn(b, t, d)
    model = APCModel(input_dim=d, hidden_dim=24, num_layers=2, offsets=(1, 3))
    preds, repr_seq = model(x)
    assert set(preds) == {1, 3}
    for n in (1, 3):
        assert preds[n].shape == (b, t, d)
    assert repr_seq.shape == (b, t, 24)


def test_apc_loss_decreases_when_fitting_shift():
    # A model trained to predict x[t+n] should drive the loss below persistence on
    # a learnable signal. Smoke-check that a few steps reduce the loss.
    torch.manual_seed(0)
    b, t, d = 8, 40, 8
    base = torch.randn(b, t, d)
    model = APCModel(input_dim=d, hidden_dim=32, num_layers=2, offsets=(2,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = last = None
    for step in range(30):
        preds, _ = model(base)
        loss, _ = apc_loss(preds, base)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first


def test_persistence_and_diagnostics_run():
    x = torch.randn(2, 20, 12)
    assert persistence_l1(x, 3) > 0
    rep = collapse_report(x)
    assert rep["feature_std"] > 0
    assert rep["effective_rank"] > 1.0
