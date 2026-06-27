import copy

import torch
import torch.nn.functional as F

from tajepa.models.actions import ActionJEPA, VectorQuantizer
from tajepa.diagnostics import codebook_perplexity


def test_vq_quantizes_and_straight_through_grad():
    vq = VectorQuantizer(num_codes=8, code_dim=4)
    e = torch.randn(2, 5, 4, requires_grad=True)
    q, idx, vq_loss, probs = vq(e)
    assert q.shape == e.shape
    assert idx.shape == (2, 5) and idx.max() < 8
    assert probs.shape == (2, 5, 8)
    # straight-through: gradient flows back to e
    q.sum().backward()
    assert e.grad is not None and e.grad.abs().sum() > 0


def test_codebook_perplexity_bounds():
    one = torch.zeros(100, dtype=torch.long)              # all one code
    assert codebook_perplexity(one, 8) < 1.01
    uni = torch.arange(8).repeat(20)                       # uniform over 8
    assert codebook_perplexity(uni, 8) > 7.5


def test_action_jepa_forward_and_indices():
    b, t, d = 2, 24, 128
    m = ActionJEPA(in_dim=d, dim=64, enc_depth=2, pred_depth=2, heads=4, num_codes=16, code_dim=16)
    x = torch.randn(b, t, d)
    out = m(x)
    assert out["z"].shape == (b, t, 64)
    assert out["pred"].shape == (b, t, 64)
    assert out["indices"].shape == (b, t)
    assert out["probs"].shape == (b, t, 16)


def test_chosen_action_codes_change_prediction():
    torch.manual_seed(0)
    b, t, d = 4, 30, 32
    m = ActionJEPA(in_dim=d, dim=32, enc_depth=2, pred_depth=2, heads=4, num_codes=8, code_dim=16)
    target = copy.deepcopy(m.encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    x = torch.randn(b, t, d)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(60):
        out = m(x)
        with torch.no_grad():
            zt = target(x)
        loss = F.smooth_l1_loss(out["pred"][:, :-1], zt[:, 1:]) + out["vq_loss"]
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for c, tp in zip(m.encoder.parameters(), target.parameters()):
                tp.mul_(0.99).add_(c, alpha=0.01)

    # different chosen codes -> different predicted next latent
    with torch.no_grad():
        codes_a = torch.zeros(b, t, dtype=torch.long)
        codes_b = torch.full((b, t), 5, dtype=torch.long)
        _, pa = m.predict_with_actions(x, codes_a)
        _, pb = m.predict_with_actions(x, codes_b)
    assert (pa - pb).abs().mean() > 1e-3
