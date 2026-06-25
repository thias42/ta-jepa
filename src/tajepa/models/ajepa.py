"""A-JEPA mel baseline — masked latent prediction over the spectrogram.

The bidirectional, X-ARES-comparable baseline (plan, Phase 0). It treats a cached
log-mel sequence ``[T, F]`` as a 2D map, splits it into time-frequency patches, and
— I-JEPA/MAE style — predicts the *latent* representations of masked patches from
the visible ones, against an EMA target encoder.

This is deliberately NOT our method: it is bidirectional (not causal) and relies on
the EMA target + stop-gradient for anti-collapse without the VICReg term that design
invariant #3 reserves for the causal JEPA. Keeping it faithful makes it a fair point
of comparison rather than a half-built version of our model. Collapse diagnostics are
still wired in at train time because the EMA target can, in principle, collapse here
too — it just isn't given the extra VICReg protection.

Note: for simplicity the predictor runs at the encoder width (I-JEPA uses a narrower
predictor); this costs a little compute but keeps positional embeddings shared and
the code legible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Positional embedding (2D sin-cos, parameter-free, supports variable grids)
# --------------------------------------------------------------------------- #
def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / (10000 ** (torch.arange(dim // 2, device=pos.device) / (dim // 2)))
    out = pos[:, None].float() * omega[None, :]
    return torch.cat([out.sin(), out.cos()], dim=1)  # [M, dim]


def sincos_2d_pos_embed(dim: int, gh: int, gw: int, device) -> torch.Tensor:
    """``[gh*gw, dim]`` row-major (h outer, w inner) to match the patch flatten order."""
    assert dim % 4 == 0, "embed dim must be divisible by 4 for 2D sincos"
    yy, xx = torch.meshgrid(
        torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij"
    )
    emb_h = _sincos_1d(dim // 2, yy.flatten())
    emb_w = _sincos_1d(dim // 2, xx.flatten())
    return torch.cat([emb_h, emb_w], dim=1)  # [gh*gw, dim]


# --------------------------------------------------------------------------- #
# Patchify + transformer stack
# --------------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """``[B, 1, F, T]`` -> ``[B, N, dim]`` patch tokens via a strided conv."""

    def __init__(self, dim: int, patch_f: int, patch_t: int) -> None:
        super().__init__()
        self.patch_f, self.patch_t = patch_f, patch_t
        self.proj = nn.Conv2d(1, dim, kernel_size=(patch_f, patch_t), stride=(patch_f, patch_t))

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(img)                       # [B, dim, gh, gw]
        gh, gw = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)         # [B, N, dim]
        return x, gh, gw


def _transformer(dim: int, depth: int, heads: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=4 * dim,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class MelEncoder(nn.Module):
    """Patch-embeds a mel map and encodes it (optionally only the visible patches)."""

    def __init__(self, dim, depth, heads, patch_f, patch_t, dropout=0.0) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(dim, patch_f, patch_t)
        self.blocks = _transformer(dim, depth, heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img, pos, ids_keep=None) -> torch.Tensor:
        x, _, _ = self.patch_embed(img)          # [B, N, dim]
        x = x + pos
        if ids_keep is not None:
            x = torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, x.shape[-1]))
        return self.norm(self.blocks(x))


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
def random_masking(B: int, N: int, ratio: float, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAE-style. Returns ``(ids_keep [B,Nv], mask [B,N] 1=masked, ids_restore [B,N])``."""
    len_keep = max(1, int(round(N * (1 - ratio))))
    noise = torch.rand(B, N, device=device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones(B, N, device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return ids_keep, mask, ids_restore


# --------------------------------------------------------------------------- #
# A-JEPA
# --------------------------------------------------------------------------- #
class AJEPA(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        predictor_depth: int = 3,
        patch_f: int = 16,
        patch_t: int = 16,
        mask_ratio: float = 0.6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_f, self.patch_t = patch_f, patch_t
        self.mask_ratio = mask_ratio
        self.encoder = MelEncoder(dim, depth, heads, patch_f, patch_t, dropout)
        self.predictor = _transformer(dim, predictor_depth, heads, dropout)
        self.predictor_norm = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def _to_img(self, feats: torch.Tensor) -> torch.Tensor:
        """``[B, T, F]`` -> ``[B, 1, F, T]``, cropped to whole patches."""
        b, t, fdim = feats.shape
        t_c = (t // self.patch_t) * self.patch_t
        f_c = (fdim // self.patch_f) * self.patch_f
        if t_c < self.patch_t or f_c < self.patch_f:
            raise ValueError(f"input {tuple(feats.shape)} too small for patch "
                             f"({self.patch_f}x{self.patch_t})")
        return feats[:, :t_c, :f_c].transpose(1, 2).unsqueeze(1)  # [B,1,F,T]

    def pos_for(self, img: torch.Tensor) -> torch.Tensor:
        gh = img.shape[-2] // self.patch_f
        gw = img.shape[-1] // self.patch_t
        return sincos_2d_pos_embed(self.dim, gh, gw, img.device).unsqueeze(0)  # [1,N,dim]

    def forward(self, feats: torch.Tensor):
        """Returns ``(pred [B,N,dim], mask [B,N], pos [1,N,dim], img)``."""
        img = self._to_img(feats)
        pos = self.pos_for(img)
        b, n = img.shape[0], pos.shape[1]
        ids_keep, mask, ids_restore = random_masking(b, n, self.mask_ratio, img.device)

        ctx = self.encoder(img, pos, ids_keep)                    # [B, Nv, dim]
        mask_tokens = self.mask_token.expand(b, n - ctx.shape[1], -1)
        x = torch.cat([ctx, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore[..., None].expand(-1, -1, self.dim))
        x = x + pos
        pred = self.predictor_norm(self.predictor(x))             # [B, N, dim]
        return pred, mask, pos, img

    @torch.no_grad()
    def encode_full(self, feats: torch.Tensor) -> torch.Tensor:
        """Full (unmasked) patch representations ``[B, N, dim]`` — for the probe."""
        img = self._to_img(feats)
        return self.encoder(img, self.pos_for(img), ids_keep=None)


def ajepa_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Smooth-L1 at masked positions against layer-normed (stop-grad) targets.

    ``target`` should already be detached (EMA encoder output). Returns ``(loss, logs)``.
    """
    target = F.layer_norm(target, (target.shape[-1],))
    m = mask.bool()
    loss = F.smooth_l1_loss(pred[m], target[m])
    return loss, {"loss": float(loss.detach()), "mask_frac": float(mask.mean())}
