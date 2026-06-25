import copy

import torch

from tajepa.models.ajepa import AJEPA, ajepa_loss, random_masking, sincos_2d_pos_embed


def test_sincos_pos_embed_shape():
    pos = sincos_2d_pos_embed(64, 5, 16, device="cpu")
    assert pos.shape == (80, 64)


def test_random_masking_counts():
    ids_keep, mask, ids_restore = random_masking(4, 80, 0.6, device="cpu")
    assert ids_keep.shape[0] == 4
    assert ids_restore.shape == (4, 80)
    # ~60% masked
    assert abs(mask.float().mean().item() - 0.6) < 0.05


def test_ajepa_forward_shapes():
    b, t, f = 2, 64, 80
    model = AJEPA(n_mels=f, dim=64, depth=2, heads=4, predictor_depth=2,
                  patch_f=16, patch_t=16, mask_ratio=0.5)
    feats = torch.randn(b, t, f)
    pred, mask, pos, img = model(feats)
    n_patches = (f // 16) * (t // 16)            # 5 * 4 = 20
    assert pred.shape == (b, n_patches, 64)
    assert mask.shape == (b, n_patches)
    assert img.shape == (b, 1, f, t)
    rep = model.encode_full(feats)
    assert rep.shape == (b, n_patches, 64)


def test_ajepa_loss_decreases_against_ema_target():
    torch.manual_seed(0)
    b, t, f = 4, 64, 80
    model = AJEPA(n_mels=f, dim=64, depth=2, heads=4, predictor_depth=2, mask_ratio=0.5)
    target = copy.deepcopy(model.encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    feats = torch.randn(b, t, f)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first = last = None
    for step in range(40):
        pred, mask, pos, img = model(feats)
        with torch.no_grad():
            tgt = target(img, pos, ids_keep=None)
        loss, _ = ajepa_loss(pred, tgt, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        # EMA the target toward the encoder.
        with torch.no_grad():
            for c, t_ in zip(model.encoder.parameters(), target.parameters()):
                t_.mul_(0.99).add_(c, alpha=0.01)
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
