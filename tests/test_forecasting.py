import copy

import torch
from torch.utils.data import Dataset

from tajepa.models.apc import APCModel
from tajepa.models.jepa import JEPA, jepa_loss, grounding_loss
from tajepa.eval import forecast_report, codec_forecast_curves


class _ToyDS(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return {"features": self.seqs[i]}


def test_forecast_report_structure_and_persistence():
    torch.manual_seed(0)
    d = 16
    model = JEPA(in_dim=d, dim=32, enc_depth=2, pred_depth=1, heads=4, offsets=(1, 2))
    target = copy.deepcopy(model.encoder)
    # A smoothly-varying signal so persistence is a meaningful (non-trivial) baseline.
    seqs = [torch.cumsum(0.1 * torch.randn(60, d), dim=0) for _ in range(4)]
    rep = forecast_report(model, target, _ToyDS(seqs), device="cpu")

    assert set(rep) == {1, 2}
    for k, m in rep.items():
        for key in ("codec_pred_cos", "codec_persist_cos", "latent_skill", "codec_l1_skill"):
            assert key in m
        # persistence cosine should be high on a slowly-varying signal, and the
        # near horizon (k=1) at least as easy to predict as the far one (k=2).
        assert -1.0 <= m["codec_persist_cos"] <= 1.0
    assert rep[1]["codec_persist_cos"] >= rep[2]["codec_persist_cos"] - 1e-6


def test_forecast_improves_after_training_toward_persistence_plus():
    # After fitting prediction + grounding on a learnable signal, the model's codec
    # cosine should beat persistence at the near horizon (positive cos-gain).
    torch.manual_seed(0)
    d = 12
    model = JEPA(in_dim=d, dim=32, enc_depth=2, pred_depth=2, heads=4, offsets=(1,))
    target = copy.deepcopy(model.encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    base = torch.sin(torch.linspace(0, 12, 80))[None, :, None] * torch.randn(1, 1, d)
    x = (base + 0.05 * torch.randn(6, 80, d))  # 6 clips, structured + noise
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(120):
        z, preds = model(x)
        with torch.no_grad():
            zt = target(x)
        loss, _ = jepa_loss(preds, z, zt, var_coef=1.0, cov_coef=0.04)
        loss = loss + grounding_loss(model.reconstruct(z), x)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for c, tp in zip(model.encoder.parameters(), target.parameters()):
                tp.mul_(0.99).add_(c, alpha=0.01)

    seqs = [x[i] for i in range(x.shape[0])]
    rep = forecast_report(model, target, _ToyDS(seqs), device="cpu")
    assert rep[1]["codec_cos_gain"] > 0


def test_codec_forecast_curves_multimodel():
    torch.manual_seed(0)
    d = 16
    jepa = JEPA(in_dim=d, dim=32, enc_depth=1, pred_depth=1, heads=4, offsets=(1, 2))
    apc = APCModel(input_dim=d, hidden_dim=32, num_layers=1, offsets=(1, 3))
    seqs = [torch.cumsum(0.1 * torch.randn(50, d), dim=0) for _ in range(5)]
    curves = codec_forecast_curves(_ToyDS(seqs), device="cpu", jepa=jepa, apc=apc, stats_clips=5)

    assert set(curves) == {"persistence", "APC", "JEPA"}
    assert set(curves["JEPA"]) == {1, 2}            # each model at its own offsets
    assert set(curves["APC"]) == {1, 3}
    assert set(curves["persistence"]) == {1, 2, 3}  # union
    for k, m in curves["persistence"].items():
        assert -1.0 <= m["cos"] <= 1.0
