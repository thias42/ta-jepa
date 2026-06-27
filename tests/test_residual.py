import copy

import numpy as np
import torch
import torch.nn.functional as F

from tajepa.models.residual import ResidualActionJEPA, _next_step_delta
from tajepa.eval import residual_action_effect_matrix, action_report


def test_next_step_delta():
    d = torch.tensor([[[0.0], [1.0], [3.0]]])     # [1,3,1]
    out = _next_step_delta(d)
    assert torch.allclose(out, torch.tensor([[[1.0], [2.0], [0.0]]]))  # last padded


def test_residual_forward_and_steering():
    b, t, d, c = 2, 24, 128, 3
    m = ResidualActionJEPA(in_dim=d, dim=64, enc_depth=2, pred_depth=2, heads=4,
                           cond_dim=c, num_codes=8, code_dim=16)
    x = torch.randn(b, t, d)
    desc = torch.randn(b, t, c)
    out = m(x, desc)
    assert out["z"].shape == (b, t, 64) and out["pred"].shape == (b, t, 64)
    assert out["indices"].shape == (b, t) and out["probs"].shape == (b, t, 8)
    # FiLM is zero-init (identity) by design; nudge both off zero to test the wiring
    # (post-training, both handles modulate the prediction).
    with torch.no_grad():
        m.predictor.film_desc.to_gb.weight.add_(0.1 * torch.randn_like(m.predictor.film_desc.to_gb.weight))
        m.predictor.film_act.to_gb.weight.add_(0.1 * torch.randn_like(m.predictor.film_act.to_gb.weight))
    z, p_codeA = m.predict_with(x, desc, torch.zeros(b, t, dtype=torch.long))
    _, p_codeB = m.predict_with(x, desc, torch.full((b, t), 5, dtype=torch.long))
    assert (p_codeA - p_codeB).abs().mean() > 1e-4                 # action code steers
    _, p_desc2 = m.predict_with(x, desc * 0, torch.zeros(b, t, dtype=torch.long))
    assert (p_codeA - p_desc2).abs().mean() > 1e-4                 # descriptor control steers


def test_residual_action_eval_runs():
    class _DS:
        def __init__(self, n, t, c):
            self.it = [{"features": torch.randn(t, 16), "control": torch.randn(t, c)} for _ in range(n)]

        def __len__(self):
            return len(self.it)

        def __getitem__(self, i):
            return self.it[i]

    m = ResidualActionJEPA(in_dim=16, dim=32, enc_depth=1, pred_depth=1, heads=4,
                           cond_dim=3, num_codes=4, code_dim=8)
    res = residual_action_effect_matrix(
        m, _DS(5, 20, 3), render_fn=lambda s: s, desc_fn=lambda a: a[..., :3],
        standardize_fn=lambda c: c, n_clips=5, device="cpu",
    )
    assert res["effect"].shape == (4, 3)
    rep = action_report(res, names=["loud", "cen", "ons"])
    assert "separability" in rep and len(rep["usage"]) == 4
