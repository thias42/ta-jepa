"""Train the APC baseline on cached frame features (codec embeddings or mel).

This is the Phase 0 known-good causal-prediction reference. It logs, every step:
the APC L1 per offset, the naive *persistence* L1 (the bar Phase 1 must beat), and
the collapse diagnostics (feature std / effective rank) so the monitoring path is
exercised from day one.

Example (after extract_embeddings.py):
    python scripts/train_apc.py \
        --cache data/cache/encodec_24khz/synthetic \
        --offsets 3 --hidden 512 --layers 3 \
        --max-steps 200 --batch-size 8 --window 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tajepa.data.embedding_dataset import EmbeddingSequenceDataset, pad_collate
from tajepa.diagnostics import collapse_report
from tajepa.models.apc import APCModel, apc_loss, persistence_l1
from tajepa.utils import seed_everything


class APCLightning(pl.LightningModule):
    def __init__(self, input_dim, hidden=512, layers=3, offsets=(3,), dropout=0.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = APCModel(input_dim, hidden, layers, tuple(offsets), dropout)
        self.offsets = tuple(offsets)
        self.lr = lr

    def training_step(self, batch, _):
        x, pad_mask = batch["features"], batch["pad_mask"]
        preds, repr_seq = self.model(x)
        loss, logs = apc_loss(preds, x, pad_mask)
        for k, v in logs.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"))
        # Reference + monitors (no grad).
        for n in self.offsets:
            self.log(f"train/persist_n{n}", persistence_l1(x, n, pad_mask))
        for k, v in collapse_report(repr_seq, pad_mask).items():
            self.log(f"diag/{k}", v, prog_bar=(k == "effective_rank"))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, nargs="+",
                    help="Dir(s) of cached .npy features (multiple = multi-domain).")
    ap.add_argument("--offsets", type=int, nargs="+", default=[3])
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--save", type=Path, default=None, help="Write final checkpoint here.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    ds = EmbeddingSequenceDataset(args.cache, window_frames=args.window)
    input_dim = ds[0]["features"].shape[-1]
    print(f"Dataset: {len(ds)} clips, feature dim = {input_dim}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        drop_last=False,
    )
    model = APCLightning(input_dim, args.hidden, args.layers, tuple(args.offsets), args.dropout, args.lr)
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        log_every_n_steps=10,
        enable_checkpointing=False,
        default_root_dir="lightning_logs/apc",
    )
    trainer.fit(model, loader)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(args.save))
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
