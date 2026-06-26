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
| **Causal JEPA (Phase 1)** | FMA, 25k steps | **44.8% ± 2.2%** ⚠ below baseline |
| **Causal JEPA + grounding** | FMA, 15k steps | **48.4% ± 3.0%** ⚠ below baseline |

## Phase 1 — causal JEPA: a clean pretext, a failing probe (gate NOT passed)

The Phase 1 causal latent JEPA **fails its own gate**, and the failure is informative.

- **Pretext objective: excellent.** Forward latent prediction crushes the persistence
  baseline at every offset (pred L1 0.07–0.12 vs persistence 0.54–1.12) and the
  representation does **not** collapse — feature std ≈ 1.04, **effective rank ≈ 241/256**.
  VICReg did exactly its job: no dimensional collapse.
- **Downstream probe: worse than doing nothing.** ESC-50 5-fold CV = **44.8% ± 2.2%**
  (45.8% with mean pooling) — ~10 points *below* the raw-codec baseline (54.7%) and ~14
  below APC (58.7%). Not a pooling artifact (checked mean vs mean+std; codec baseline
  reproduces at 54.4% through the same code path).

**Diagnosis — representation/prediction decoupling.** The encoder is free to map codec
embeddings into *any* latent that is (a) predictable by the causal predictor and (b)
high-variance/decorrelated (VICReg). It found such a space — predictable and full-rank —
but one that discards class-discriminative acoustic detail present in the raw codec.
Contrast APC (58.7%): APC regresses the *actual* future codec frame, so its hidden state
must stay acoustically rich; our JEPA predicts a *moving EMA target in a free latent
space*, where online encoder and target co-adapt toward an easily-predictable manifold
with no anchor to the input. VICReg prevents *dimensional* collapse but does not guarantee
the dimensions carry useful *semantics*. This is precisely the risk the plan's Phase 1
gate is designed to catch — and it caught it.

**Next levers (Phase 1 iteration — "the real work").** Most promising first:
1. **Anchor the latent to the codec space** — an auxiliary `z_t → x_t` (or `x_{t+1}`)
   reconstruction term, i.e. give the JEPA APC's grounding so the latent must stay
   acoustically informative. Most direct test of the diagnosis.
2. **Full-context (non-causal) target encoder** (I-JEPA/V-JEPA style) instead of the
   causal EMA copy, so targets are richer and less co-adaptive.
3. Sweep the VICReg balance / loss (cosine vs smooth-L1) and the offset set; probe
   earlier checkpoints to test whether *more* pretext training actively erodes the probe.

Per the plan, **do not proceed to Phase 2 (control) until a Phase 1 variant beats the
APC bar (58.7%) and persistence.** Beating persistence alone (done) is not sufficient.

### Iteration #1 — latent grounding (helps, not enough) + a pooling insight

Adding the `z_t → codec-frame` reconstruction anchor lifted the probe **44.8% → 48.4%**
(meanstd), and the head reconstructs the (standardized) codec frame near-perfectly
(recon MSE ≈ 0.014) — so `z` *linearly contains* the codec embedding. Still below the
54.7% baseline. Breaking the probe down by pooling is the revealing part:

| Representation | mean-pool CV | mean+std-pool CV |
|---|---|---|
| EnCodec embeddings (baseline) | 41.2% | 54.4% |
| Causal JEPA + grounding | **45.7%** | 48.4% |

**The JEPA's per-frame representation (mean-pool) beats the codec baseline (45.7 vs 41.2).**
Its deficit is entirely in the *temporal-std* component: raw codec gains +13 pts from
std-pooling, the JEPA only +2.7. The causal-predictive objective makes `z` **temporally
smooth** (that's what "predictable over time" means) — which is exactly the property a
world model wants, but it erases the frame-to-frame variability that std-pooling exploits
for environmental-sound classification.

So "fails the gate" is more precisely: *the linear probe with std-pooling rewards
temporal variability that our smooth latent deliberately suppresses.* This raises a real
question about whether meanstd-probe accuracy on ESC-50 is the right yardstick for a world
model, alongside the model-side levers (less aggressive smoothing via near-offset
weighting / shorter horizons; stronger or different grounding; attentive pooling).

**Open question for the project owner:** treat this as (a) a model problem — push for a
less-smooth latent that wins on meanstd too; (b) a metric problem — the world-model
objective and the std-pooling probe are partly at odds, so add predictive/forecasting and
(later) control-based evals rather than leaning on the probe alone; or (c) both.

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
