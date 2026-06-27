"""Phase 2a+2b — train the residual-action JEPA.

Descriptor deltas (loudness/brightness) condition the predictor for free; the learned VQ
action is pushed onto the residual transition. Trains on codec + descriptor caches, with
the same EMA target + VICReg + grounding + VQ + code-entropy as the action JEPA.

    python scripts/train_residual.py \
        --features data/cache/encodec_24khz/fma_small \
        --control  data/cache/descriptors/fma_small \
        --num-codes 16 --code-dim 32 --max-steps 25000 --save runs/residual.ckpt
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import _bootstrap  # noqa: F401

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tajepa.data.embedding_dataset import PairedSequenceDataset, pad_collate
from tajepa.diagnostics import collapse_report, codebook_perplexity
from tajepa.models.residual import ResidualActionJEPA
from tajepa.models.jepa import vicreg_terms, grounding_loss, latent_persistence_l1
from tajepa.utils import seed_everything


def standardize(c, pad_mask):
    valid = c[~pad_mask] if pad_mask is not None else c.reshape(-1, c.shape[-1])
    mu, sd = valid.mean(0), valid.std(0).clamp_min(1e-4)
    return (c - mu) / sd


class ResidualLightning(pl.LightningModule):
    def __init__(self, in_dim=128, cond_dim=3, dim=256, enc_depth=6, pred_depth=3, heads=4,
                 num_codes=16, code_dim=32, commitment_cost=0.25, dropout=0.0,
                 lr=2e-4, weight_decay=0.05, var_coef=1.0, cov_coef=0.04, vq_coef=1.0,
                 grounding_coef=1.0, entropy_coef=0.1, base_momentum=0.996, max_steps=25000):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResidualActionJEPA(in_dim, dim, enc_depth, pred_depth, heads, cond_dim,
                                        num_codes, code_dim, commitment_cost, dropout)
        self.target = copy.deepcopy(self.model.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.num_codes = num_codes
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
        desc = standardize(batch["control"], pad)
        out = self.model(x, desc, pad)
        z, pred = out["z"], out["pred"]
        with torch.no_grad():
            z_tgt = self.target(x, pad)

        valid = (~pad[:, 1:]).unsqueeze(-1) if pad is not None else None
        diff = F.smooth_l1_loss(pred[:, :-1], z_tgt[:, 1:], reduction="none")
        pred_loss = (diff * valid).sum() / (valid.sum().clamp(min=1) * z.shape[-1]) \
            if valid is not None else diff.mean()

        var_loss, cov_loss = vicreg_terms(z, pad)
        recon_loss = grounding_loss(self.model.reconstruct(z), x, pad)
        avg_probs = out["probs"][~pad].mean(0) if pad is not None else out["probs"].reshape(-1, self.num_codes).mean(0)
        code_entropy = -(avg_probs * (avg_probs + 1e-9).log()).sum()

        loss = (pred_loss + self.hparams.vq_coef * out["vq_loss"]
                + self.hparams.var_coef * var_loss + self.hparams.cov_coef * cov_loss
                + self.hparams.grounding_coef * recon_loss
                - self.hparams.entropy_coef * code_entropy)

        self.log("train/loss", float(loss.detach()), prog_bar=True)
        self.log("train/pred_loss", float(pred_loss.detach()))
        self.log("train/vq_loss", float(out["vq_loss"].detach()))
        self.log("train/persist_l1", latent_persistence_l1(z_tgt, 1, pad))
        self.log("diag/perplexity", codebook_perplexity(out["indices"], self.num_codes), prog_bar=True)
        for k, v in collapse_report(z, pad).items():
            self.log(f"diag/{k}", v)
        return loss

    def on_train_batch_end(self, *args):
        self._ema_update(self._momentum())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True, nargs="+", help="Codec cache dir(s).")
    ap.add_argument("--control", type=Path, required=True, nargs="+", help="Descriptor cache dir(s).")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--pred-depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--num-codes", type=int, default=16)
    ap.add_argument("--code-dim", type=int, default=32)
    ap.add_argument("--commitment-cost", type=float, default=0.25)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--var-coef", type=float, default=1.0)
    ap.add_argument("--cov-coef", type=float, default=0.04)
    ap.add_argument("--vq-coef", type=float, default=1.0)
    ap.add_argument("--grounding-coef", type=float, default=1.0)
    ap.add_argument("--entropy-coef", type=float, default=0.1)
    ap.add_argument("--base-momentum", type=float, default=0.996)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=25000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    feats = args.features if len(args.features) > 1 else args.features[0]
    ctrl = args.control if len(args.control) > 1 else args.control[0]
    ds = PairedSequenceDataset(feats, ctrl, window_frames=args.window)
    in_dim = ds[0]["features"].shape[-1]
    cond_dim = ds[0]["control"].shape[-1]
    print(f"Dataset: {len(ds)} clips, in_dim={in_dim}, cond_dim={cond_dim}, codes={args.num_codes}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=pad_collate, drop_last=True)
    model = ResidualLightning(
        in_dim=in_dim, cond_dim=cond_dim, dim=args.dim, enc_depth=args.enc_depth,
        pred_depth=args.pred_depth, heads=args.heads, num_codes=args.num_codes,
        code_dim=args.code_dim, commitment_cost=args.commitment_cost, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay, var_coef=args.var_coef, cov_coef=args.cov_coef,
        vq_coef=args.vq_coef, grounding_coef=args.grounding_coef, entropy_coef=args.entropy_coef,
        base_momentum=args.base_momentum, max_steps=args.max_steps)
    trainer = pl.Trainer(max_steps=args.max_steps, accelerator=args.accelerator,
                         log_every_n_steps=10, enable_checkpointing=False,
                         default_root_dir="lightning_logs/residual")
    trainer.fit(model, loader)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(args.save))
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
