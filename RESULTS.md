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

### Cross-model curves — persistence vs APC vs JEPA (codec space)

`codec_forecast_curves` puts all three on the same axes (globally-standardized codec space,
so they're directly comparable; cosine of predicted vs true future frame). APC predicts
codec frames directly (the strong reference); JEPA decodes its predicted latent; persistence
is the codec baseline. Each model at its own trained offsets. (Note: this uses *global*
standardization, vs the *per-clip* standardization in the single-model table above — hence
slightly different absolute cosines; conclusions agree.)

**ESC-50 (transfer)** — cosine, with gain over persistence:

| k | persistence | APC | JEPA |
|---|---|---|---|
| 1 | 0.565 | 0.671 (**+0.107**) | 0.516 (−0.049) |
| 3 | 0.531 | 0.639 (**+0.107**) | — |
| 4 | 0.515 | — | 0.465 (−0.050) |
| 8 | 0.490 | — | 0.448 (−0.043) |

**FMA (in-domain)** — cosine, with gain over persistence:

| k | persistence | APC | JEPA |
|---|---|---|---|
| 1 | 0.398 | 0.582 (**+0.184**) | 0.540 (**+0.142**) |
| 2 | 0.353 | — | 0.479 (**+0.126**) |
| 3 | 0.333 | 0.484 (**+0.151**) | — |
| 4 | 0.313 | — | 0.440 (**+0.127**) |
| 8 | 0.284 | — | 0.412 (**+0.128**) |

JEPA latent-space skill (own space): ESC-50 +17/+34/+45/+41%, FMA +13/+23/+32/+30%
(k=1/2/4/8) — strongly positive in *both* domains.

**What the cross-model benchmark shows:**

1. **In-domain, the JEPA is a genuine forecaster** — it beats persistence at every horizon
   (+0.13 cosine, flat across k), competitive with APC and *more stable* over the horizon
   (APC is only defined at 1/3/5). This vindicates the model on the metric a world model is
   actually for, after the probe (the wrong yardstick) made it look broken.
2. **The transfer gap is the decoder, not the predictor.** On ESC-50, APC's frame-forecast
   still beats persistence (+0.11; "predict something like recent frames" transfers), while
   JEPA's *decoded* forecast falls below (−0.05). But the JEPA's *latent* skill stays strongly
   positive on ESC-50 (+17–45%) — so the encoder's dynamics-prediction transfers fine; only
   the music-trained grounding-head decoder doesn't. → broadening training data targets
   exactly this (the domain-dependent part), consistent with the earlier conclusion.
3. The probe deficit (separate, metric-driven) and this forecasting-transfer gap
   (decoder/data) are **two different issues** — and on the forecasting metric, in-domain,
   the JEPA is not deficient at all.

### Broadening to general audio closes the ESC-50 transfer gap (multi-domain)

The diagnosis predicted the ESC-50 codec-forecasting gap was the *music-trained decoder*
failing to generalize — not a model defect. Test: pretrain a multi-domain JEPA on FMA
(music) **+ FSD50K** (~41k general sound-event clips) and re-run the forecasting eval. It
closed the gap.

| ESC-50 forecasting | FMA-only | **FMA + FSD50K** |
|---|---|---|
| JEPA codec cos gain, k=1 | −0.049 | **+0.086** |
| ... k=2 | −0.050 | **+0.100** |
| ... k=4 | −0.050 | **+0.107** |
| ... k=8 | −0.043 | **+0.100** |
| JEPA latent skill, k=1/2/4/8 | +17/34/45/41% | **+29/46/51/45%** |

(Multi-domain JEPA: 25k steps, effective_rank 226/256, no collapse. Trained on Modal.)

The multi-domain JEPA now **beats persistence at every horizon on unseen environmental
sound** (+0.09–0.11 cosine) and essentially **matches APC** — the codec-frame specialist —
on this transfer domain (APC +0.09/+0.10/+0.10 at k=1/3/5), while winning decisively in its
native latent space (+29–51%, up from +17–45%). General-audio pretraining both fixed the
decoder's transfer *and* improved latent forecasting.

This confirms the whole chain: the transfer gap was decoder/data, not a model defect; and on
the metric a world model is actually for, Phase 1 is now **strong** — beats persistence,
matches the specialist baseline on transfer, wins in latent space. (The meanstd-ESC-50
*probe* remains a separate, unchanged metric artifact — temporal smoothness, see above.)

Next extensions: multi-step rollout skill (Phase 3); decoded-audio listening tests; and an
in-domain forecasting check on the multi-domain model (FSD50K-eval / FMA val).

### Is the probe deficit domain mismatch? (No.) — in-domain control

We pretrain on FMA (music) and the main probe is ESC-50 (environmental), so a natural worry
is that the JEPA's probe deficit is just cross-domain transfer. Cheap control: probe the
same FMA-pretrained JEPA *in-domain* on FMA genre (it has genre labels), vs codec.

| Representation | FMA genre (in-domain) | ESC-50 (transfer) |
|---|---|---|
| codec baseline | 42.0% | 54.4% |
| grounded JEPA | 31.1% | 48.4% |

(FMA: 2500-train/800-test single split, 8 genres, meanstd — not CV, but internally
consistent.) The JEPA underperforms codec **in-domain too**, by an even *larger* margin
(−11 pts vs −6 on ESC-50). So the probe deficit is **not** domain mismatch — it is the
temporal-smoothness/metric effect, present in both domains. **Broadening training data
would not fix the probe number.**

Two distinct things, don't conflate them:
- **Probe/representation-quality deficit** → a *metric* effect (smooth latent vs std-pooling),
  domain-independent. Fix: better evals (this section's direction), not more data.
- **Codec-space forecasting transfer** (in-domain FMA positive, ESC-50 negative) → a *decoder*
  generalization effect that *is* domain-dependent. More-diverse training data would help
  this — and is the plan's intent anyway (general audio, not music-only; we trained FMA-only
  for compute). So broaden data to serve the general-audio thesis and forecasting transfer,
  **not** as a probe fix.

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

## Phase 2a — supervised control: closed-loop controllability

First real controllability result. Controllable JEPA trained on FMA + FSD50K (25k steps,
effective rank 225/256, no collapse), evaluated closed-loop on ESC-50 (offset 1, bump +2σ,
100 clips): perturb one descriptor's delta, render the prediction to audio via the EnCodec
decoder, re-extract the descriptors. Matrix is column-normalized (each measured descriptor
by its largest driver):

| perturb \ measure | loudness | centroid | onset |
|---|---|---|---|
| **loudness** | **+1.00** | +0.38 | −1.00 |
| **centroid** | +0.52 | **+1.00** | −0.49 |
| **onset** | +0.003 | +0.002 | +0.000 |

diagonal all positive · diagonal-dominant 0.67 · dominance ratio 1.67.

- **Loudness and brightness (centroid) are controllable** — each most strongly drives its
  own measured descriptor, with modest cross-talk. The dials are honest.
- **The onset dial is dead** — the onset *row* is ~0: perturbing it changes nothing in the
  rendered audio. The model's FiLM for onset stayed near its zero init, i.e. the predictor
  found the onset delta unusable and ignored it. (Measured onset is instead driven,
  negatively, by loudness — a rendering/acoustic side effect.)

**Why, and what it implies.** Loudness and centroid vary smoothly, so their one-frame delta
is predictable and informative — the predictor leans on it. Onset strength is a noisy
temporal-derivative feature whose 1-frame delta is nearly unpredictable, so it carried no
usable control signal. This lines up with the design thesis: supervised descriptors for the
smooth *envelope* axes; **learned latent actions (2b) for transients / texture / percussion**
— onset/transients look like a 2b job, not a 2a one. (Caveat: this is offset 1 only; an
offset sweep is the cheap next check before declaring onset fundamentally uncontrollable.)

## Phase 2b — learned latent actions: the codebook rediscovers loudness

Action JEPA trained on FMA + FSD50K (25k steps, codebook perplexity 12.1/16, no collapse).
Actions-controllability eval on ESC-50 (40 clips, force each code → render → re-extract MIR;
effect vs the inferred-action baseline). Representative codes:

| code | loudness | centroid | onset | usage | consistency |
|---|---|---|---|---|---|
| 11 | **+1.73** | +0.06 | −0.12 | 8.9% | 0.77 |
| 1 | **−1.29** | −0.03 | +0.10 | 10.9% | 0.84 |
| 5 | **−0.98** | −0.01 | +0.07 | 7.8% | 0.81 |
| 3 | **+1.00** | +0.02 | −0.09 | 5.5% | 0.72 |
| 13 | **−0.64** | +0.00 | +0.03 | 9.4% | 0.73 |

mean consistency 0.65 · separability 0.92 · **16/16 codes used** · every code's top descriptor = loudness.

**Finding.** The codes are consistent and all used, but they collapsed onto a *single semantic
axis — loudness*: the 16 codes form a loudness spectrum (≈ −1.3 … +1.7), while centroid/onset
effects are ~10× smaller and onset only trails loudness (an anti-correlated side effect, not
independent control). So the learned action vocabulary **rediscovered the dominant,
easily-predictable transition (energy)** — the very axis 2a's descriptors already control —
rather than the transient/texture/onset control we hoped 2b would unlock. (Not "leakage": the
control is real and reusable, just redundant and 1-dimensional.)

**Why.** With a limited code budget, the inverse model spends it on the most prediction-relevant
axis — frame-to-frame energy. The onset/texture structure is lower-energy and got crowded out.

**Next — combine 2a + 2b (residual actions).** Feed the *supervised* descriptor deltas
(loudness, centroid) as known control, and have the *learned* actions explain only the
**residual** transition. That removes loudness from the actions' job and forces the codebook
onto non-loudness structure (transients/texture) — the natural union of the plan's two control
paths, and the most promising route to the onset control neither path has cracked alone.

## Phase 2a+2b — residual actions: the codebook starts to capture transients

Pure 2b's codebook collapsed to loudness. The residual model gives the predictor the
descriptor delta for free (FiLM) and keeps the learned action on a small codebook, to push
the codes onto the residual. Trained on FMA + FSD50K (25k steps); residual actions eval on
ESC-50 (40 clips, descriptors held fixed, only the code varied):

| | pure 2b actions | **2a+2b residual** |
|---|---|---|
| codes with dominant effect = loudness | 16 / 16 | **12 / 16** |
| codes with dominant effect = **onset** | 0 / 16 | **4 / 16** |
| mean consistency | 0.65 | 0.60 |
| separability | 0.92 | 0.87 |

**The needle moved.** Freeing loudness to the descriptor path let **4 codes become
onset-dominant** (vs none in pure 2b) — directional confirmation that the residual split
pushes the codebook onto the transient structure descriptors can't express.

**But transient control is still the hard frontier.** The onset codes are weak (effects
~0.08, and all slightly *negative*) and *inconsistent* (consistency ~0.3 vs ~0.7 for the
loudness codes); 12 codes still track loudness, so descriptors didn't fully absorb it. So
this is progress, not a clean transient dial.

**The through-line of Phase 2.** Across all three control paths, the *smooth envelope* axes
(loudness, brightness) are easy and reliable; *transients/onset* are the consistent hard
axis — 2a's onset dial was dead, 2b ignored onset entirely, and 2a+2b finally captures it
but weakly. Closing that gap (stronger descriptor absorption / explicit two-stage residual /
a transient-friendlier target than per-frame onset) is the open Phase 2 problem.

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
