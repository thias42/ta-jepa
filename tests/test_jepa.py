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
        loss = grounding_loss(model.reconstruct(z), x,
                              mean=model.codec_mean, std=model.codec_std)
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


def test_context_window_is_recorded_and_survives_a_checkpoint():
    m = JEPA(in_dim=8, dim=16, enc_depth=1, pred_depth=1, heads=2, offsets=(1,))
    assert m.trained_context is None            # unknown until training records it
    m.set_context_frames(256)
    assert m.trained_context == 256
    m2 = JEPA(in_dim=8, dim=16, enc_depth=1, pred_depth=1, heads=2, offsets=(1,))
    m2.load_state_dict(m.state_dict())
    assert m2.trained_context == 256


def test_windowed_predict_covers_every_frame_with_in_range_positions():
    """Long inputs must never be run in one pass: absolute positional encodings are
    defined past the trained length but were never learned there, so a single pass
    silently measures extrapolation."""
    import copy

    from tajepa.models.jepa import windowed_predict

    torch.manual_seed(0)
    m = JEPA(in_dim=8, dim=16, enc_depth=1, pred_depth=1, heads=2, offsets=(1, 2)).eval()
    tgt = copy.deepcopy(m.encoder).eval()
    for t in (40, 64, 65, 200, 375):
        preds, z = windowed_predict(m, tgt, torch.randn(1, t, 8), context=64, stride=32)
        assert z.shape == (1, t, 16)
        assert torch.isfinite(z).all() and (z.abs().sum(-1) > 0).all(), f"gap at T={t}"
        for o in (1, 2):
            assert preds[o].shape == (1, t, 16)
            assert torch.isfinite(preds[o]).all()


def test_windowed_predict_matches_a_single_pass_when_it_fits():
    import copy

    from tajepa.models.jepa import windowed_predict

    torch.manual_seed(0)
    m = JEPA(in_dim=8, dim=16, enc_depth=1, pred_depth=1, heads=2, offsets=(1,)).eval()
    tgt = copy.deepcopy(m.encoder).eval()
    x = torch.randn(1, 50, 8)
    preds, z = windowed_predict(m, tgt, x, context=64, stride=32)
    with torch.no_grad():
        _, direct = m(x)
    assert torch.allclose(preds[1], direct[1], atol=1e-5)
    assert torch.allclose(z, tgt(x), atol=1e-5)


def test_windowed_frames_use_only_in_range_positions():
    """The whole point: no frame is ever produced at a position >= context."""
    import copy

    from tajepa.models.jepa import windowed_predict

    seen = []
    real = JEPA.forward

    def spy(self, x, pad_mask=None):
        seen.append(x.shape[1])
        return real(self, x, pad_mask)

    torch.manual_seed(0)
    m = JEPA(in_dim=8, dim=16, enc_depth=1, pred_depth=1, heads=2, offsets=(1,)).eval()
    tgt = copy.deepcopy(m.encoder).eval()
    JEPA.forward = spy
    try:
        windowed_predict(m, tgt, torch.randn(1, 400, 8), context=64, stride=32)
    finally:
        JEPA.forward = real
    assert seen and max(seen) <= 64, f"a window exceeded the context: {max(seen)}"
