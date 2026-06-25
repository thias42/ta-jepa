import copy

import torch

from tajepa.models.jepa import (
    JEPA,
    CausalTransformer,
    jepa_loss,
    vicreg_terms,
    grounding_loss,
    latent_persistence_l1,
)


def test_jepa_forward_shapes():
    b, t, d_in = 2, 32, 128
    model = JEPA(in_dim=d_in, dim=64, enc_depth=2, pred_depth=2, heads=4, offsets=(1, 3))
    x = torch.randn(b, t, d_in)
    z, preds = model(x)
    assert z.shape == (b, t, 64)
    assert set(preds) == {1, 3}
    for o in (1, 3):
        assert preds[o].shape == (b, t, 64)


def test_encoder_is_causal():
    # Changing inputs at position k onward must not change encoder outputs before k.
    torch.manual_seed(0)
    enc = CausalTransformer(in_dim=16, dim=32, depth=3, heads=4).eval()
    x = torch.randn(1, 20, 16)
    with torch.no_grad():
        z1 = enc(x)
        x2 = x.clone()
        x2[:, 10:] = torch.randn_like(x2[:, 10:])
        z2 = enc(x2)
    assert torch.allclose(z1[:, :10], z2[:, :10], atol=1e-5)
    assert not torch.allclose(z1[:, 10:], z2[:, 10:], atol=1e-5)


def test_vicreg_variance_penalizes_collapse():
    collapsed = torch.zeros(4, 10, 8)              # constant -> std 0 -> high var loss
    diverse = torch.randn(4, 10, 8) * 2.0
    var_c, _ = vicreg_terms(collapsed)
    var_d, _ = vicreg_terms(diverse)
    assert var_c > var_d
    assert var_c > 0.9                              # near the hinge target gamma=1


def test_grounding_reconstruction_learns():
    # The grounding head should be able to reconstruct a standardized codec frame from
    # the latent of a clip that is a deterministic function of its input.
    torch.manual_seed(0)
    b, t, d = 8, 20, 16
    model = JEPA(in_dim=d, dim=32, enc_depth=2, pred_depth=1, heads=4, offsets=(1,))
    x = torch.randn(b, t, d)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = last = None
    for step in range(60):
        z = model.encode(x)
        loss = grounding_loss(model.reconstruct(z), x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
    assert model.reconstruct(model.encode(x)).shape == (b, t, d)


def test_persistence_and_loss_run():
    b, t, d = 2, 20, 32
    model = JEPA(in_dim=d, dim=32, enc_depth=2, pred_depth=1, heads=4, offsets=(2,))
    x = torch.randn(b, t, d)
    z, preds = model(x)
    tgt = torch.randn(b, t, 32)
    loss, logs = jepa_loss(preds, z, tgt, var_coef=1.0, cov_coef=0.04)
    assert "var_loss" in logs and "pred_loss" in logs
    assert torch.isfinite(loss)
    assert latent_persistence_l1(tgt, 2) > 0


def test_training_step_does_not_collapse():
    # With an EMA target + VICReg, the online std should not crater to ~0.
    torch.manual_seed(0)
    b, t, d = 8, 24, 32
    model = JEPA(in_dim=d, dim=32, enc_depth=2, pred_depth=2, heads=4, offsets=(1, 2))
    target = copy.deepcopy(model.encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    x = torch.randn(b, t, d)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(40):
        z, preds = model(x)
        with torch.no_grad():
            z_tgt = target(x)
        loss, logs = jepa_loss(preds, z, z_tgt, var_coef=1.0, cov_coef=0.04)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for c, tp in zip(model.encoder.parameters(), target.parameters()):
                tp.mul_(0.99).add_(c, alpha=0.01)
    with torch.no_grad():
        std = model.encoder(x).std(dim=(0, 1)).mean().item()
    assert std > 0.3            # not collapsed to a constant
