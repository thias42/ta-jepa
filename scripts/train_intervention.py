"""Phase 2 — train the action-conditioned JEPA on applied interventions.

The observed stream is the *intervened* audio (identical to clean before the event), the
conditioning is the commanded action summed over each prediction window, and the target is
the intervened future. So the model is asked the world-model question directly: given what
I have heard and what is about to be done, what will I hear?

This is `ControllableJEPA` with the action vector in place of the descriptor delta — the
Phase 2a machinery survives; what changes is that the conditioning is now *exogenous*.
Two deliberate differences from Phase 2a:

- **The encoder does not see the action** (`augment_input` stays off). Descriptor work fed
  controls to the encoder so the latent could represent them; an action is not an observation
  and feeding it in would re-create the circularity the whole redesign is meant to remove.
- **The action cannot be shortcut.** It is zero except across a transition, and at that
  transition it is the only information about what is coming — so a model that ignores it
  cannot predict the change. There is no descriptor-delta leak to exploit.

Prediction loss is logged separately on event frames and elsewhere: away from an event the
action is zero and the task reduces to Phase 1, so the aggregate loss hides whether the
action is used at all.

    python scripts/train_intervention.py --cache data/cache/interventions/fma_small \
        --offsets 1 2 4 8 --max-steps 25000 --save runs/intervention.ckpt
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tajepa.data.embedding_dataset import InterventionPairDataset, pad_collate
from tajepa.diagnostics import collapse_report
from tajepa.interventions import AXES
from tajepa.models.action_conditioning import action_windows
from tajepa.models.control import ControllableJEPA
from tajepa.models.jepa import backfill_codec_stats, grounding_loss, vicreg_terms
from tajepa.utils import seed_everything


class InterventionLightning(pl.LightningModule):
    def __init__(self, in_dim=128, cond_dim=3, dim=256, enc_depth=6, pred_depth=3, heads=4,
                 offsets=(1, 2, 4, 8), dropout=0.0, lr=2e-4, weight_decay=0.05,
                 var_coef=1.0, cov_coef=0.04, grounding_coef=1.0,
                 base_momentum=0.996, max_steps=25000):
        super().__init__()
        self.save_hyperparameters()
        self.model = ControllableJEPA(in_dim, dim, enc_depth, pred_depth, heads,
                                      tuple(offsets), cond_dim, dropout, augment_input=False)
        self.target = copy.deepcopy(self.model.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.offsets = tuple(offsets)
        # Action scale, filled from the data before fit: raw units (dB, octaves, seconds)
        # differ by orders of magnitude, so an unscaled FiLM would be driven by gain alone.
        self.register_buffer("action_std", torch.ones(cond_dim))

    def on_load_checkpoint(self, checkpoint) -> None:
        backfill_codec_stats(checkpoint["state_dict"], "model.", self.hparams.in_dim)

    def _momentum(self) -> float:
        p = min(1.0, self.global_step / max(1, self.hparams.max_steps))
        return 1.0 - (1.0 - self.hparams.base_momentum) * (math.cos(math.pi * p) + 1) / 2

    @torch.no_grad()
    def _ema(self, m: float) -> None:
        for c, t in zip(self.model.encoder.parameters(), self.target.parameters()):
            t.mul_(m).add_(c.detach(), alpha=1 - m)

    def training_step(self, batch, _):
        x, pad = batch["intervened"], batch["pad_mask"]      # the stream actually observed
        act = batch["action"] / self.action_std
        deltas = action_windows(act, self.offsets)

        z = self.model.encoder(x, pad)
        preds = self.model.predictor(z, deltas, pad)
        self.target.eval()
        with torch.no_grad():
            z_tgt = self.target(x, pad).detach()

        pred_loss, n = z.new_zeros(()), 0
        ev_num = ev_den = qu_num = qu_den = 0.0
        for o in self.offsets:
            if x.shape[1] <= o:
                continue
            p, tgt = preds[o][:, :-o], z_tgt[:, o:]
            valid = (~pad[:, o:]).unsqueeze(-1)
            err = torch.nn.functional.smooth_l1_loss(p, tgt, reduction="none") * valid
            pred_loss = pred_loss + err.sum() / (valid.sum().clamp(min=1) * p.shape[-1])
            n += 1
            with torch.no_grad():   # split by whether an action spans this window
                ev = (deltas[o][:, :-o].abs().sum(-1, keepdim=True) > 0) & valid
                ev_num += float((err * ev).sum()); ev_den += float(ev.sum() * p.shape[-1])
                qu = valid & ~ev
                qu_num += float((err * qu).sum()); qu_den += float(qu.sum() * p.shape[-1])
        pred_loss = pred_loss / max(1, n)

        var_l, cov_l = vicreg_terms(z, pad)
        recon = grounding_loss(self.model.reconstruct(z), x, pad,
                               mean=self.model.codec_mean, std=self.model.codec_std)
        loss = (pred_loss + self.hparams.var_coef * var_l
                + self.hparams.cov_coef * cov_l + self.hparams.grounding_coef * recon)

        self.log("train/loss", float(loss.detach()), prog_bar=True)
        self.log("train/pred_loss", float(pred_loss.detach()))
        self.log("train/recon_loss", float(recon.detach()))
        self.log("train/var_loss", float(var_l.detach()))
        # the diagnostic that matters: the aggregate is dominated by action-free frames
        if ev_den > 0:
            self.log("train/pred_at_event", ev_num / ev_den, prog_bar=True)
        if qu_den > 0:
            self.log("train/pred_no_action", qu_num / qu_den)
        for k, v in collapse_report(z, pad).items():
            self.log(f"diag/{k}", v, prog_bar=(k == "effective_rank"))
        return loss

    def on_train_batch_end(self, *args):
        self._ema(self._momentum())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="Intervention pair cache dir.")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--pred-depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grounding-coef", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=25000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    ds = InterventionPairDataset(args.cache, window_frames=args.window)
    in_dim = ds[0]["features"].shape[-1]
    cond_dim = ds[0]["action"].shape[-1]
    print(f"Dataset: {len(ds)} pairs, in_dim={in_dim}, action axes={AXES}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, collate_fn=pad_collate)
    model = InterventionLightning(
        in_dim=in_dim, cond_dim=cond_dim, dim=args.dim, enc_depth=args.enc_depth,
        pred_depth=args.pred_depth, heads=args.heads, offsets=tuple(args.offsets),
        lr=args.lr, grounding_coef=args.grounding_coef, max_steps=args.max_steps)

    # Codec statistics for the grounding head's output space (see data/stats.py), and the
    # per-axis action scale — raw units differ by orders of magnitude (dB vs octaves vs
    # seconds), so without this the FiLM would be driven by the gain axis alone.
    sample = [ds[i] for i in range(0, len(ds), max(1, len(ds) // 200))][:200]
    frames = torch.cat([s["features"] for s in sample], 0)
    model.model.set_codec_stats(frames.mean(0), frames.std(0).clamp_min(1e-4))
    model.model.set_context_frames(args.window)
    acts = torch.cat([s["action"] for s in sample], 0)
    scale = acts.abs().sum(0) / (acts.abs() > 0).sum(0).clamp(min=1)
    model.action_std.copy_(scale.clamp_min(1e-3))
    print(f"Action scale (mean |delta| per axis): "
          f"{ {n: round(float(v), 3) for n, v in zip(AXES, model.action_std)} }")

    trainer = pl.Trainer(max_steps=args.max_steps, accelerator=args.accelerator,
                         log_every_n_steps=10, enable_checkpointing=False,
                         default_root_dir="lightning_logs/intervention")
    trainer.fit(model, loader)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(args.save))
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
