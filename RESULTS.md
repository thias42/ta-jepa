# Results — Phase 0 baselines & Phase 1 causal JEPA

Results for the temporally-controlled audio JEPA. Phase 0 validates the pipeline and
sets the baselines; Phase 1 is the core causal world model. See
`docs/temporal-audio-jepa-plan.md` for the plan and `CLAUDE.md` for invariants. New to
the project? Start with **In plain English** below, and see the **Glossary** at the end.

_Last updated: 2026-06-26._

## In plain English (for a non-specialist)

**What we're building.** A system that learns about sound the way you might learn about
the world by paying attention: by constantly trying to *predict what comes next*. Play it
audio and it learns to guess the near future of that sound. The bet is that to predict
sound well, it has to genuinely "understand" it — instruments, footsteps, a car
approaching — without ever being told labels. (Eventually we also want to *steer* the
prediction, like imagining "what if it got louder / higher" — but that's later.)

**How we check if it learned something useful.** We freeze the model and give it a simple
pop quiz: show it 2,000 short clips of everyday sounds (dogs, rain, helicopters, ...) and
see whether a *very* simple classifier can read off the right category from the model's
internal representation. Better understanding → easier to read off → higher score. We
compare against two reference points: the raw audio codec with no learning at all, and a
simpler "predict the next frame" model (APC).

**What happened.** The core model learned to *predict the future* beautifully — far better
than a dumb "assume nothing changes" baseline — and its internal representation stayed rich
and healthy (it didn't cheat by collapsing to a constant). But on the pop quiz it scored
*lower* than the no-learning baseline. Surprising.

**Why — the interesting part.** We dug in and found the model's understanding is genuinely
good *moment to moment* (frame by frame it matches the APC model). The catch: the quiz, the
way it's scored, secretly rewards representations that **jump around a lot from instant to
instant**. Our model, precisely because it learned to predict the future, represents sound
in a **smooth, steady** way — and we *measured* this directly: its internal signal changes
~2.4× more gently over time than the raw audio. That smoothness is a *feature* of a good
predictor, not a bug. The quiz just happens to grade on jumpiness.

**The takeaway.** This is like judging a student who deeply understands a story's plot
(can tell you what happens next) with a quiz that only rewards noticing flashy
scene-to-scene cuts. The student isn't worse — the quiz measures the wrong thing. So rather
than dumbing the model down to win a flawed quiz, the right move is to **build a better
test** — one that measures what a world model is actually for: predicting and (later)
controlling sound over time. That's the current decision.

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

| Representation | mean-pool CV | mean+std-pool CV | std gain |
|---|---|---|---|
| EnCodec embeddings (baseline) | 41.2% | 54.4% | +13.2 |
| APC (FMA) | 46.8% | 58.7% | +11.9 |
| Causal JEPA + grounding | **45.7%** | 48.4% | **+2.7** |

**The JEPA's per-frame representation (mean-pool) essentially matches APC (45.7 vs 46.8)
and beats the codec baseline (41.2).** The entire APC→JEPA gap on meanstd is the
*temporal-std* component: codec/APC gain +12–13 pts from std-pooling, the JEPA only +2.7.

**Mechanism verified directly (not just inferred).** We measured the temporal smoothness of
each representation on ESC-50 (200 clips):

| Representation | lag-1 autocorrelation (1 = smooth) | norm. frame-to-frame step (low = smooth) |
|---|---|---|
| codec | 0.277 | 0.980 |
| APC hidden | 0.270 | 0.990 |
| JEPA + grounding | **0.669** | **0.774** |

The JEPA latent is **~2.4× more temporally autocorrelated** than codec/APC, with smaller
frame-to-frame steps — it is genuinely, measurably smoother. APC's hidden state is exactly
as jumpy as the raw codec, which is why std-pooling works for it. So the chain is solid:
causal-predictive objective → smooth latent → std-pooling extracts little → low meanstd
score, despite per-frame quality matching APC.

### Decision: fix the evaluation, not the model

The std-pooling probe rewards instant-to-instant variability — a property a *world model*
should legitimately suppress, since temporal predictability is the whole point. Chasing a
jumpier latent to win this probe would be optimizing the wrong objective. So the agreed
direction is to **build evaluations that measure what a world model is actually for**, e.g.:

- **Forecasting quality** — latent (and decoded) prediction error vs horizon, vs persistence
  and APC, on held-out audio (the model already wins this; formalize and report it properly).
- **Probes that don't bake in a jumpiness prior** — e.g. attentive/learned temporal pooling,
  or evaluating per-frame then aggregating, rather than mean+std.
- **(Later) control-based evals** — perturb a control signal, decode, re-measure, check the
  intended attribute changed and others didn't (the plan's closed-loop controllability test).

Until such evals exist, "below the codec baseline on meanstd-ESC-50" is noted but **not**
treated as the verdict on Phase 1.

## Forecasting-error-vs-horizon (world-model evaluation)

The first of the better evals (`src/tajepa/eval/forecasting.py`, `scripts/run_forecast.py`).
It asks the question a *world model* is actually for: **does it predict the future of the
audio, better than assuming nothing changes (persistence)?** — at each horizon `k`. We
measure both in the model's own latent space and, decoding the predicted latent back to
codec space via the grounding head, in codec space (cosine + standardized L1). Everything
is reported as *skill vs persistence*, so a temporally-smooth latent gets no free pass.

Grounded JEPA (FMA-pretrained), 400 clips per set:

| | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| **latent skill** (own space) — ESC-50 | +17% | +34% | +45% | +42% |
| **latent skill** — FMA (in-domain) | +13% | +23% | +32% | +29% |
| **codec L1 skill** — ESC-50 (transfer) | −25% | −18% | −12% | −7% |
| **codec L1 skill** — FMA (in-domain) | **+11%** | **+14%** | **+16%** | **+18%** |
| codec persistence cosine (how predictable codec is) | ~0.3 | ~0.2 | ~0.2 | ~0.15 |

What it shows:

- **The model has real predictive skill.** In its own latent space it beats persistence at
  every horizon (+17–45%), strongest at medium horizons — it learned dynamics, not just
  smoothness (persistence is the smoothness-aware baseline).
- **In-domain, it forecasts the actual future audio** (codec space) better than persistence
  (+11–18% L1). On the **ESC-50 transfer** domain it does not — the music-trained
  decoder/predictor doesn't generalize to environmental sound (though latent skill stays
  positive there, so the encoder's dynamics-prediction does transfer; the decoder is the
  weak link).
- **Codec frames are intrinsically hard to predict** — adjacent-frame cosine is only ~0.3
  (consistent with the 0.27 autocorr): the codec embedding has jumpy fine structure. This
  is precisely the "codec-token unpredictability" the design sidesteps by predicting in a
  **smoother latent** rather than reconstructing codec frames — so the result *validates*
  the core design choice rather than indicting it.

**In plain English:** the model is a decent fortune-teller for what a sound will "be like"
a moment from now (its internal forecast beats just guessing "same as now"), and on the kind
of audio it studied (music) it can even forecast the raw audio fingerprint. It struggles to
forecast the raw fingerprint of *unfamiliar* sounds — but that raw fingerprint is jittery
and nearly unpredictable anyway, which is the whole reason we predict the smoothed "gist"
instead.

Next extensions of this eval: add APC and the codec baseline as comparison curves (APC
forecasts codec frames directly, so it should be the strong codec-space reference);
multi-step rollout skill (Phase 3); and decoded-audio listening tests.

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
predictor + VICReg) must beat. As of the temporal-smoothness finding above, criterion (b)
— the meanstd-ESC-50 probe — is under review as a yardstick for a world model; see the
"fix the evaluation" decision.

## Glossary

Terms and abbreviations used in this doc and the project.

**Models & methods**
- **JEPA** — Joint-Embedding Predictive Architecture. Learns by predicting the
  *representation* (embedding) of unseen data from visible data, rather than the raw pixels/
  samples. Our model is a *causal* JEPA over audio.
- **World model** — a model that learns the dynamics of its input so it can predict (and
  eventually be steered to imagine) future states.
- **APC** — Autoregressive Predictive Coding. Baseline that predicts the *actual* future
  feature frame from past context (an LSTM here). Cannot collapse (the target is grounded).
- **A-JEPA** — audio JEPA baseline: *bidirectional* masked latent prediction over the
  spectrogram (predict masked patches' latents). Not causal; no VICReg.
- **I-JEPA / V-JEPA** — image / video JEPA (Meta). The lineage this project extends to
  causal, controllable audio.
- **Predictor** (`g_φ`) — small network that maps the encoder's latents to predicted future
  latents.
- **Frame encoder** (`f_θ`) — the network that turns the input sequence into latents `z`.
- **Grounding / reconstruction anchor** — auxiliary term forcing the latent to be able to
  reconstruct the input codec frame, so it stays acoustically informative.

**Training & anti-collapse**
- **EMA** — Exponential Moving Average. The *target encoder* `f_θ̄` is a slowly-updated copy
  of the online encoder; it provides stable prediction targets.
- **Stop-gradient** — targets are detached so gradients don't flow into the target branch.
- **Collapse** — failure mode where the encoder outputs a (near-)constant, making prediction
  trivially perfect but the representation useless.
- **VICReg** — Variance-Invariance-Covariance Regularization. Anti-collapse term: a
  *variance* hinge keeps each dimension's spread up; a *covariance* term decorrelates
  dimensions. Mandatory here (the EMA target can collapse).
- **Effective rank** — entropy-based count of "effectively used" representation dimensions;
  a live collapse monitor (drops toward 1 under collapse). Ours stays ~240/256.
- **Persistence baseline** — the naive predictor "the future equals the present"
  (`x[t+k] := x[t]`); the floor a real predictor must beat.
- **Offset / horizon (`k`, `n`)** — how many frames ahead we predict; we predict several
  offsets jointly.
- **Causal** — position `t` may only attend to positions `≤ t` (no peeking ahead),
  enforced by a causal attention mask.
- **Momentum (EMA)** — the EMA decay (e.g. 0.996→1.0 on a schedule); higher = slower target.

**Evaluation**
- **Linear probe** — freeze the model, train only a linear classifier on its features; tests
  how *linearly readable* the learned representation is.
- **Pooling (mean / mean+std)** — collapsing a `[T, D]` sequence to one vector per clip:
  `mean` over time, optionally concatenated with the `std` over time. The std component
  rewards temporal variability.
- **Standardization (z-score)** — per-feature mean-0/std-1 rescaling before the linear probe.
- **CV / fold** — k-fold cross-validation; ESC-50 ships 5 official folds. We report
  mean ± std across folds.
- **X-ARES** — an audio representation evaluation suite (the plan's probe reference).
- **Chance level** — accuracy of random guessing (1/50 = 2% on ESC-50).
- **Cross-domain transfer** — pretrain on one domain (music/FMA), evaluate on another
  (environmental/ESC-50).
- **Autocorrelation (lag-1)** — correlation of a signal with itself one step later; ~1 means
  temporally smooth. Used to verify the smoothness of the JEPA latent.

**Data & audio**
- **Codec / EnCodec / DAC** — neural audio codecs. We use EnCodec's **continuous,
  pre-quantizer** encoder embeddings (before the discrete vector quantizer).
- **Embedding / latent (`z`)** — a learned vector representation of audio.
- **Frame / frame rate** — codec/mel produce one feature vector per short time step; here
  75 frames/second.
- **Mel / log-mel spectrogram** — a perceptually-spaced time-frequency representation; the
  A-JEPA baseline's input.
- **Patch / mask ratio** — A-JEPA splits the spectrogram into 2D patches and masks a fraction
  (the mask ratio) to predict.
- **FMA / FMA-small** — Free Music Archive; 8000-track music set used for pretraining.
- **ESC-50** — Environmental Sound Classification, 2000 clips / 50 classes; eval set.
- **AudioSet / MTG-Jamendo / UrbanSound8K** — other audio datasets named in the plan (not
  yet wired).

**Infra**
- **MPS** — Apple Metal Performance Shaders (the Mac GPU backend for PyTorch).
- **Checkpoint (`.ckpt`)** — saved model weights.
- **`dim` / `depth` / `heads`** — transformer width / number of layers / attention heads.
