import copy

import torch
from torch.utils.data import Dataset

from tajepa.models.apc import APCModel
from tajepa.models.jepa import JEPA, jepa_loss, grounding_loss
from tajepa.eval import forecast_report, codec_forecast_curves
from tajepa.models.linear_ar import fit_linear_ar


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
    # As in real training: record the codec statistics once, then ground against them.
    flat = x.reshape(-1, d)
    model.set_codec_stats(flat.mean(0), flat.std(0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(120):
        z, preds = model(x)
        with torch.no_grad():
            zt = target(x)
        loss, _ = jepa_loss(preds, z, zt, var_coef=1.0, cov_coef=0.04)
        loss = loss + grounding_loss(model.reconstruct(z), x,
                                     mean=model.codec_mean, std=model.codec_std)
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
    curves = codec_forecast_curves(_ToyDS(seqs), device="cpu", jepa=jepa, apc=apc)

    assert set(curves) == {"persistence", "APC", "JEPA"}
    assert set(curves["JEPA"]) == {1, 2}            # each model at its own offsets
    assert set(curves["APC"]) == {1, 3}
    assert set(curves["persistence"]) == {1, 2, 3}  # union
    for k, m in curves["persistence"].items():
        assert -1.0 <= m["cos"] <= 1.0


def test_ar_baseline_is_reported_and_beats_persistence():
    # The AR floor must appear in the curves and, on an autoregressive signal, be a
    # strictly harder reference than persistence — that is the point of retiring it.
    torch.manual_seed(0)
    d = 8
    w = torch.randn(d, d) * 0.25
    seqs = []
    for _ in range(12):
        z = torch.randn(120, d)
        for t in range(1, 120):
            z[t] = z[t - 1] @ w + 0.1 * z[t]
        seqs.append(z)
    ds = _ToyDS(seqs)
    ar = fit_linear_ar(ds, dim=d, offsets=(1, 2), order=4, max_clips=12)
    jepa = JEPA(in_dim=d, dim=16, enc_depth=1, pred_depth=1, heads=4, offsets=(1, 2))
    curves = codec_forecast_curves(ds, device="cpu", jepa=jepa, ar=ar)

    assert "AR(4)" in curves
    for k in (1, 2):
        assert curves["AR(4)"][k]["cos"] > curves["persistence"][k]["cos"]


def test_forecast_report_scores_against_ar_when_supplied():
    torch.manual_seed(0)
    d = 8
    model = JEPA(in_dim=d, dim=16, enc_depth=1, pred_depth=1, heads=4, offsets=(1,))
    target = copy.deepcopy(model.encoder)
    seqs = [torch.cumsum(0.1 * torch.randn(60, d), dim=0) for _ in range(6)]
    ds = _ToyDS(seqs)
    ar = fit_linear_ar(ds, dim=d, offsets=(1,), order=4, max_clips=6)
    rep = forecast_report(model, target, ds, device="cpu", codec_ar=ar)
    for key in ("codec_ar_cos", "codec_cos_gain_vs_ar", "codec_l1_skill_vs_ar"):
        assert key in rep[1]
    # AR skill is measured against the AR reference, not against persistence
    assert rep[1]["codec_cos_gain_vs_ar"] == (rep[1]["codec_pred_cos"] - rep[1]["codec_ar_cos"])


def test_eval_depends_only_on_raw_output_not_on_the_heads_parameterization():
    """The bug this eval was rewritten to prevent.

    Two models that emit *identical* raw codec but record different statistics (the head
    absorbing the difference) are the same predictor and must score identically. The old
    path compared the head's standardized output against the eval set's statistics, so
    changing that internal split moved the score — which is how a music-trained model
    scored on ESC-50 looked like it had failed to transfer.
    """
    torch.manual_seed(0)
    d = 8
    a = JEPA(in_dim=d, dim=16, enc_depth=1, pred_depth=1, heads=4, offsets=(1,))
    mean, std = torch.full((d,), 5.0), torch.full((d,), 3.0)

    b = copy.deepcopy(a)
    b.set_codec_stats(mean, std)
    with torch.no_grad():   # reparameterize so reconstruct_raw is unchanged
        b.recon_head.weight.div_(std[:, None])
        b.recon_head.bias.copy_((a.recon_head.bias - mean) / std)

    seqs = [torch.cumsum(0.1 * torch.randn(60, d), dim=0) + 5.0 for _ in range(5)]
    ds = _ToyDS(seqs)
    z = a.encode(seqs[0].unsqueeze(0))
    assert torch.allclose(a.reconstruct_raw(z), b.reconstruct_raw(z), atol=1e-5)
    assert not torch.allclose(a.reconstruct(z), b.reconstruct(z), atol=1e-3)

    ca = codec_forecast_curves(ds, device="cpu", jepa=a)
    cb = codec_forecast_curves(ds, device="cpu", jepa=b)
    assert abs(ca["JEPA"][1]["cos"] - cb["JEPA"][1]["cos"]) < 1e-4
    assert abs(ca["JEPA"][1]["l1"] - cb["JEPA"][1]["l1"]) < 1e-4
