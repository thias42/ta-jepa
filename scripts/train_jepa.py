"""Phase 1 — train the causal latent JEPA on cached codec embeddings.

The core model (models/jepa.py): causal frame encoder + EMA target + causal predictor
over multiple future offsets, with latent smooth-L1 prediction + VICReg variance/
covariance. Every step logs the prediction loss, the VICReg terms, the EMA momentum,
the forward latent-prediction L1 vs a persistence baseline (the gate: must beat
persistence), and the collapse diagnostics on the online latents.

Example:
    python scripts/train_jepa.py --cache data/cache/encodec_24khz/fma_small \
        --dim 256 --enc-depth 6 --pred-depth 3 --offsets 1 2 3 4 \
        --window 256 --batch-size 32 --max-steps 20000 --accelerator mps \
        --save runs/jepa.ckpt
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
from tajepa.models.jepa import JEPA, jepa_loss, grounding_loss, latent_persistence_l1
from tajepa.utils import seed_everything


class JEPALightning(pl.LightningModule):
    def __init__(
        self, in_dim=128, dim=256, enc_depth=6, pred_depth=3, heads=4,
        offsets=(1, 2, 3, 4), dropout=0.0, lr=2e-4, weight_decay=0.05,
        var_coef=1.0, cov_coef=0.04, grounding_coef=0.0,
        base_momentum=0.996, max_steps=20000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.jepa = JEPA(in_dim, dim, enc_depth, pred_depth, heads, tuple(offsets), dropout)
        self.target = copy.deepcopy(self.jepa.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.offsets = tuple(offsets)
        self.base_momentum = base_momentum
        self.max_steps_ = max_steps

    def _momentum(self) -> float:
        progress = min(1.0, self.global_step / max(1, self.max_steps_))
        return 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * progress) + 1) / 2

    @torch.no_grad()
    def _ema_update(self, m: float) -> None:
        for c, t in zip(self.jepa.encoder.parameters(), self.target.parameters()):
            t.mul_(m).add_(c.detach(), alpha=1 - m)

    def training_step(self, batch, _):
        x, pad = batch["features"], batch["pad_mask"]
        z, preds = self.jepa(x, pad)
        self.target.eval()
        with torch.no_grad():
            z_tgt = self.target(x, pad)
        loss, logs = jepa_loss(preds, z, z_tgt, pad,
                               var_coef=self.hparams.var_coef, cov_coef=self.hparams.cov_coef)
        if self.hparams.grounding_coef > 0:
            recon_loss = grounding_loss(self.jepa.reconstruct(z), x, pad)
            loss = loss + self.hparams.grounding_coef * recon_loss
            self.log("train/recon_loss", float(recon_loss.detach()))

        for k in ("loss", "pred_loss", "var_loss", "cov_loss"):
            self.log(f"train/{k}", logs[k], prog_bar=(k == "loss"))
        self.log("train/ema_m", self._momentum())
        # Forward latent-prediction vs persistence (the gate).
        for o in self.offsets:
            self.log(f"train/pred_l1_n{o}", logs.get(f"pred_l1_n{o}", float("nan")))
            self.log(f"train/persist_l1_n{o}", latent_persistence_l1(z_tgt, o, pad))
        for k, v in collapse_report(z, pad).items():
            self.log(f"diag/{k}", v, prog_bar=(k == "effective_rank"))
        return loss

    def on_train_batch_end(self, *args):
        self._ema_update(self._momentum())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.jepa.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, nargs="+",
                    help="Dir(s) of cached codec .npy (multiple = multi-domain pretraining).")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--pred-depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--var-coef", type=float, default=1.0)
    ap.add_argument("--cov-coef", type=float, default=0.04)
    ap.add_argument("--grounding-coef", type=float, default=0.0,
                    help="Weight on the z_t->codec-frame reconstruction anchor (0=off).")
    ap.add_argument("--base-momentum", type=float, default=0.996)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    ds = EmbeddingSequenceDataset(args.cache, window_frames=args.window,
                                  min_frames=max(args.offsets) + 1)
    in_dim = ds[0]["features"].shape[-1]
    print(f"Dataset: {len(ds)} clips, in_dim = {in_dim}, window = {args.window}, "
          f"offsets = {args.offsets}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=pad_collate, drop_last=True)
    model = JEPALightning(
        in_dim=in_dim, dim=args.dim, enc_depth=args.enc_depth, pred_depth=args.pred_depth,
        heads=args.heads, offsets=tuple(args.offsets), dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, var_coef=args.var_coef, cov_coef=args.cov_coef,
        grounding_coef=args.grounding_coef,
        base_momentum=args.base_momentum, max_steps=args.max_steps,
    )
    trainer = pl.Trainer(
        max_steps=args.max_steps, accelerator=args.accelerator, log_every_n_steps=10,
        enable_checkpointing=False, default_root_dir="lightning_logs/jepa",
    )
    trainer.fit(model, loader)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(args.save))
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
