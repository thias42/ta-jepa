# Phase 0 results

Baselines for the temporally-controlled audio JEPA. These exist to (a) validate the
full pipeline end-to-end on real data and (b) set the bar that the Phase 1 causal
JEPA must clear. See `docs/temporal-audio-jepa-plan.md` for the plan and `CLAUDE.md`
for invariants.

_Last updated: 2026-06-25._

## Setup

- **Frontend:** EnCodec 24 kHz continuous pre-quantizer embeddings (75 Hz, dim 128);
  log-mel (80 bins, 75 Hz) for the A-JEPA baseline.
- **Pretraining data:** FMA-small (7994 music tracks, 30 s).
- **Eval data:** ESC-50 (2000 clips, 50 environmental classes, official 5 folds).
- **Hardware:** Apple MPS (pretraining) / CPU (probe). Conda env `ta-jepa`, Python 3.11.

This is a **cross-domain transfer** test by construction: representations are
pretrained on music and evaluated on environmental sound.

## Protocol (representation quality)

Frozen representation → mean(/mean+std)-pool over time → linear probe. We report the
proper ESC-50 protocol: **leave-one-fold-out 5-fold CV**, each fold's linear probe
averaged over **3 seeds**, reported as **mean ± std across folds**. (Std is the honest
error bar; earlier single-split numbers lacked one and ran ~7 pts lower because the
probe saw fewer training clips.)

## Results — ESC-50 linear probe (50-way, chance 2.0%)

| Representation | Pretraining | 5-fold CV acc |
|---|---|---|
| EnCodec embeddings (no pretraining) | — | **54.7% ± 2.6%** |
| APC (causal, codec embeddings) | FMA, 2.5k steps | **58.7% ± 3.1%** |
| A-JEPA (bidirectional, mel) | FMA, 15k steps | **55.6% ± 3.3%** |
| A-JEPA (bidirectional, mel) | FMA, 40k steps | **57.5% ± 2.5%** |

Per-fold (CV) for reference:

| | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|
| codec | 0.533 | 0.515 | 0.569 | 0.584 | 0.533 |
| APC | 0.598 | 0.548 | 0.617 | 0.621 | 0.553 |
| A-JEPA 15k | 0.546 | 0.569 | 0.529 | 0.613 | 0.523 |
| A-JEPA 40k | 0.542 | 0.589 | 0.572 | 0.615 | 0.558 |

## Predictive quality (sanity)

APC beats a naive persistence baseline (predict `x[t+n] := x[t]`) on held-out codec
embeddings — e.g. on ESC-50, APC L1 ≈ 0.70 vs persistence ≈ 0.89 at offset 3. This is
the plan's "beats persistence" gate for a causal predictor, met by the APC reference.

## Reading these numbers

- **APC (causal) shows a ~4-point lift over the raw-codec baseline**, consistent across
  folds — a mildly encouraging signal for the project's causal-prediction thesis. The
  gap is on the order of the fold std, so it is suggestive, not significant.
- **A-JEPA improves with training** — 55.6% (15k) → 57.5% (40k), confirming the 15k run
  was under-trained and that the trend is still positive (more steps would likely help
  further). At 40k it clears the baseline by ~3 points but stays below APC. Do not read
  "APC > A-JEPA" as a finding yet — different modality (codec vs mel), different training
  budget, single runs.
- The headline is methodological: the complete loop (multi-domain audio → cached
  features → causal *and* bidirectional pretraining → one shared CV probe → comparable
  numbers with error bars) runs on full-size data without collapse.

## Reproduce

```bash
P=$(conda run -n ta-jepa which python)
# Baselines pretrained on FMA (see README for the data-prep + extract steps):
$P scripts/train_apc.py   --cache data/cache/encodec_24khz/fma_small --offsets 1 3 5 \
    --hidden 512 --layers 3 --max-steps 2500 --save runs/apc_fma.ckpt
$P scripts/train_ajepa.py --cache data/cache/logmel/fma_small --dim 256 --depth 6 \
    --mask-ratio 0.6 --max-steps 15000 --accelerator mps --save runs/ajepa_fma.ckpt

# 5-fold CV probe on ESC-50 (codec baseline / APC / A-JEPA):
$P scripts/run_probe.py --manifest data/manifests/esc50.jsonl \
    --cache data/cache/encodec_24khz/esc50 --representation codec --cv --device cpu
$P scripts/run_probe.py --manifest data/manifests/esc50.jsonl \
    --cache data/cache/encodec_24khz/esc50 --representation apc \
    --apc-ckpt runs/apc_fma.ckpt --cv --device cpu
$P scripts/run_probe.py --manifest data/manifests/esc50.jsonl \
    --cache data/cache/logmel/esc50 --representation ajepa \
    --ajepa-ckpt runs/ajepa_fma.ckpt --pool mean --cv --device cpu
```

## The Phase 1 gate

The plan gates Phase 1 on the causal backbone (a) beating persistence and (b) being
X-ARES-competitive. The APC reference meets (a) and probes competitively at **58.7%**,
which is the concrete bar the Phase 1 causal JEPA (frame encoder + EMA target + causal
predictor + VICReg) must beat.
