"""Train the A-JEPA mel baseline on cached log-mel sequences.

Masked latent prediction with an EMA target encoder (no VICReg — see models/ajepa.py
for why this baseline stays faithful to I-JEPA/A-JEPA rather than borrowing our
causal model's anti-collapse term). Logs the prediction loss, the EMA momentum, and
the collapse diagnostics on the *target* representations every step.

Example (after extract_mel.py):
    python scripts/train_ajepa.py \
        --cache data/cache/logmel/fma_small \
        --dim 256 --depth 6 --mask-ratio 0.6 \
        --window 256 --batch-size 16 --max-steps 2000 --save runs/ajepa.ckpt
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import _bootstrap  # noqa: F401

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tajepa.data.embedding_dataset import EmbeddingSequenceDataset, pad_collate
from tajepa.diagnostics import collapse_report
from tajepa.models.ajepa import AJEPA, ajepa_loss
from tajepa.utils import seed_everything


class AJEPALightning(pl.LightningModule):
    def __init__(
        self, n_mels=80, dim=256, depth=6, heads=4, predictor_depth=3,
        patch_f=16, patch_t=16, mask_ratio=0.6, dropout=0.0,
        lr=1.5e-4, weight_decay=0.05, base_momentum=0.996, max_steps=2000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ajepa = AJEPA(n_mels, dim, depth, heads, predictor_depth,
                           patch_f, patch_t, mask_ratio, dropout)
        # EMA target encoder: a frozen copy of the context encoder.
        self.target = copy.deepcopy(self.ajepa.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.base_momentum = base_momentum
        self.max_steps_ = max_steps

    def _momentum(self) -> float:
        progress = min(1.0, self.global_step / max(1, self.max_steps_))
        return 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * progress) + 1) / 2

    @torch.no_grad()
    def _ema_update(self, m: float) -> None:
        for ctx_p, tgt_p in zip(self.ajepa.encoder.parameters(), self.target.parameters()):
            tgt_p.mul_(m).add_(ctx_p.detach(), alpha=1 - m)

    def training_step(self, batch, _):
        feats = batch["features"]                      # [B, T, F]
        pred, mask, pos, img = self.ajepa(feats)
        self.target.eval()
        with torch.no_grad():
            target = self.target(img, pos, ids_keep=None)   # [B, N, dim]
        loss, logs = ajepa_loss(pred, target, mask)

        self.log("train/loss", logs["loss"], prog_bar=True)
        self.log("train/mask_frac", logs["mask_frac"])
        self.log("train/ema_m", self._momentum())
        for k, v in collapse_report(target).items():
            self.log(f"diag/{k}", v, prog_bar=(k == "effective_rank"))
        return loss

    def on_train_batch_end(self, *args):
        self._ema_update(self._momentum())

    def configure_optimizers(self):
        # Only the context encoder + predictor are optimized; target follows by EMA.
        return torch.optim.AdamW(
            self.ajepa.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, nargs="+",
                    help="Dir(s) of cached log-mel .npy (multiple = multi-domain).")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--predictor-depth", type=int, default=3)
    ap.add_argument("--patch-f", type=int, default=16)
    ap.add_argument("--patch-t", type=int, default=16)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--base-momentum", type=float, default=0.996)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    ds = EmbeddingSequenceDataset(args.cache, window_frames=args.window, min_frames=args.patch_t)
    n_mels = ds[0]["features"].shape[-1]
    print(f"Dataset: {len(ds)} clips, n_mels = {n_mels}, window = {args.window}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=pad_collate, drop_last=True)
    model = AJEPALightning(
        n_mels=n_mels, dim=args.dim, depth=args.depth, heads=args.heads,
        predictor_depth=args.predictor_depth, patch_f=args.patch_f, patch_t=args.patch_t,
        mask_ratio=args.mask_ratio, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, base_momentum=args.base_momentum, max_steps=args.max_steps,
    )
    trainer = pl.Trainer(
        max_steps=args.max_steps, accelerator=args.accelerator, log_every_n_steps=10,
        enable_checkpointing=False, default_root_dir="lightning_logs/ajepa",
    )
    trainer.fit(model, loader)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(args.save))
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
