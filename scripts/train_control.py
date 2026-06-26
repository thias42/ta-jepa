"""Phase 2a — train the controllable JEPA (descriptor-delta FiLM conditioning).

Same causal encoder + EMA target + VICReg as Phase 1, but the predictor is conditioned on
the delta of frame-aligned MIR descriptors (loudness / brightness / onset / ...), so it
learns control as transition modulation. Descriptors are standardized per batch so the
deltas are balanced; at inference you set the deltas to steer.

Example (after extract_embeddings.py + extract_descriptors.py):
    python scripts/train_control.py \
        --features data/cache/encodec_24khz/esc50 \
        --control  data/cache/descriptors/esc50 \
        --dim 256 --offsets 1 2 4 8 --grounding-coef 1.0 --max-steps 20000 --save runs/control.ckpt
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

from tajepa.data.embedding_dataset import PairedSequenceDataset, pad_collate
from tajepa.diagnostics import collapse_report
from tajepa.models.control import ControllableJEPA
from tajepa.models.jepa import jepa_loss, grounding_loss, latent_persistence_l1
from tajepa.utils import seed_everything


def standardize(c: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Per-batch z-score of the control descriptors over valid frames."""
    valid = c[~pad_mask] if pad_mask is not None else c.reshape(-1, c.shape[-1])
    mu, sd = valid.mean(0), valid.std(0).clamp_min(1e-4)
    return (c - mu) / sd


class ControlLightning(pl.LightningModule):
    def __init__(self, in_dim=128, cond_dim=3, dim=256, enc_depth=6, pred_depth=3, heads=4,
                 offsets=(1, 2, 3, 4), dropout=0.0, lr=2e-4, weight_decay=0.05,
                 var_coef=1.0, cov_coef=0.04, grounding_coef=1.0,
                 base_momentum=0.996, max_steps=20000):
        super().__init__()
        self.save_hyperparameters()
        self.model = ControllableJEPA(in_dim, dim, enc_depth, pred_depth, heads,
                                      tuple(offsets), cond_dim, dropout)
        self.target = copy.deepcopy(self.model.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.offsets = tuple(offsets)
        self.base_momentum = base_momentum
        self.max_steps_ = max_steps

    def _momentum(self):
        progress = min(1.0, self.global_step / max(1, self.max_steps_))
        return 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * progress) + 1) / 2

    @torch.no_grad()
    def _ema_update(self, m):
        for c, t in zip(self.model.encoder.parameters(), self.target.parameters()):
            t.mul_(m).add_(c.detach(), alpha=1 - m)

    def training_step(self, batch, _):
        x, pad = batch["features"], batch["pad_mask"]
        ctrl = standardize(batch["control"], pad)
        z, preds = self.model(x, ctrl, pad)
        self.target.eval()
        with torch.no_grad():
            z_tgt = self.target(x, pad)
        loss, logs = jepa_loss(preds, z, z_tgt, pad,
                               var_coef=self.hparams.var_coef, cov_coef=self.hparams.cov_coef)
        if self.hparams.grounding_coef > 0:
            recon_loss = grounding_loss(self.model.reconstruct(z), x, pad)
            loss = loss + self.hparams.grounding_coef * recon_loss
            self.log("train/recon_loss", float(recon_loss.detach()))

        for k in ("loss", "pred_loss", "var_loss", "cov_loss"):
            self.log(f"train/{k}", logs[k], prog_bar=(k == "loss"))
        self.log("train/ema_m", self._momentum())
        for o in self.offsets:
            self.log(f"train/persist_l1_n{o}", latent_persistence_l1(z_tgt, o, pad))
        for k, v in collapse_report(z, pad).items():
            self.log(f"diag/{k}", v, prog_bar=(k == "effective_rank"))
        return loss

    def on_train_batch_end(self, *args):
        self._ema_update(self._momentum())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True, nargs="+", help="Codec cache dir(s).")
    ap.add_argument("--control", type=Path, required=True, help="Descriptor cache dir.")
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
    ap.add_argument("--grounding-coef", type=float, default=1.0)
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
    feature_dirs = args.features if len(args.features) > 1 else args.features[0]
    ds = PairedSequenceDataset(feature_dirs, args.control, window_frames=args.window)
    in_dim = ds[0]["features"].shape[-1]
    cond_dim = ds[0]["control"].shape[-1]
    print(f"Dataset: {len(ds)} clips, in_dim={in_dim}, cond_dim={cond_dim}, offsets={args.offsets}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=pad_collate, drop_last=True)
    model = ControlLightning(
        in_dim=in_dim, cond_dim=cond_dim, dim=args.dim, enc_depth=args.enc_depth,
        pred_depth=args.pred_depth, heads=args.heads, offsets=tuple(args.offsets),
        dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
        var_coef=args.var_coef, cov_coef=args.cov_coef, grounding_coef=args.grounding_coef,
        base_momentum=args.base_momentum, max_steps=args.max_steps,
    )
    trainer = pl.Trainer(max_steps=args.max_steps, accelerator=args.accelerator,
                         log_every_n_steps=10, enable_checkpointing=False,
                         default_root_dir="lightning_logs/control")
    trainer.fit(model, loader)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(args.save))
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
