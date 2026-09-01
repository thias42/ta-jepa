# Temporally-Controlled General-Purpose Audio JEPA — Implementation Plan

A causal, action-conditioned latent world model for general audio (music, environmental sound, speech), not a speech-only model. The goal is to predict *future* audio representations in embedding space and to steer that prediction with control signals — i.e. the audio analogue of an action-conditioned V-JEPA, not just a static representation learner.

---

## Findings vs plan (as of 2026-06-28)

This document is the original design rationale; the body below is preserved as written. Execution
(Phases 0–2) revised several points — see `RESULTS.md` for the data behind each:

- **The Phase 1 gate is not passed, and the substitute gate was not a bar.** The probe
  criterion: the JEPA probes *below* the codec baseline (44.8% vs 54.7%). That is largely a
  *readout* effect — the latent is temporally smooth (autocorr 0.67 vs 0.27) and std-pooling
  cannot read it; routing the latent through the grounding head before pooling recovers most
  of the gap (53.8% vs 55.0%). Two caveats on the story as first told: an *untrained* causal
  transformer already measures 0.58 autocorr, so the smoothness is mostly architectural
  rather than earned by the objective; and X-ARES itself was never run, only a homemade
  ESC-50 probe. The substituted gate — "beats persistence on forecasting" — is cleared by a
  closed-form **linear AR(4)**. Scored *inside* the 256-frame training window the predictor
  beats AR(4) at every horizon (latent +3% to +10%, codec +0.010 to +0.036) — but only by an
  order of magnitude less than its +17–45% over persistence. Scored beyond that window it
  collapses below persistence, because the absolute sinusoidal positional encodings were
  never learned there; that is a real limitation and a separate one. The
  forecasting eval also scored the grounding head in a space it never emitted into, which
  manufactured the "music-trained decoder doesn't transfer" finding (sign flips once fixed)
  — which is what the multi-domain campaign was launched to close. Re-run in-context, the
  FMA+FSD50K checkpoint beats AR(4) at every horizon (+0.023 to +0.027 codec, +6.6% to
  +10.5% latent) and is the only one of the three models to clear that floor on ESC-50
  transfer. Multi-domain data did not improve the AR-relative number over FMA-only
  (both ~+7-10%); it moved only the persistence-relative one (+34% to +48%). Both defects are fixed in code; see
  the Correction section of `RESULTS.md`. Open experiment: horizons well beyond the
  13–107 ms tested, where linear extrapolation fails.
- **Control axis selection (#4 / Phase 2a).** What predicts controllability is
  **loudness-decorrelation + codec-recoverability**, not the descriptor's name. The plan's
  **onset/transient** axis is *render-limited* and does NOT control (codec-recoverability R²≈0.07);
  **harmonic_ratio** (tonal-vs-noisy; not in the plan's list, but matching its "envelope-vs-fine-
  structure" instinct) *does* (R²≈0.34). Working dials: **loudness, brightness, harmonic_ratio**;
  pitch weak; transients open. 
- **The decoder is not peripheral (#1/#6/Phase 4, and the "no-decoder" novelty framing).** A
  grounding head (latent→codec decoder) was added to the core, and the *linear* render path is the
  bottleneck that blocks transient control — the render stage is central to controllability, not
  optional. It also carries more weight than acknowledged: it is what makes the representation
  recoverable under pooling, and its output space is load-bearing enough that mis-recording it
  inverted a headline result. Untested premise behind the "nonlinear decoder" fix: whether
  transients survive a *clean* EnCodec round-trip at all (encode true frames → decode →
  re-extract). If they don't, the limit is the codec, not the head.
- **Learned actions (#4b / Phase 2b).** Implemented as predictor **FiLM-conditioning, not the
  "mid-stack" VQ insertion**; and contrary to the "pretext loss worsens" expectation, the action
  makes prediction *easier* (leak risk) and the codebook **collapsed to a loudness axis**. The
  residual 2a+2b variant only weakly captured transients.
- **Data/stack divergences:** used **FSD50K** as the general set (not AudioSet, impractical to
  download); **librosa** only (not madmom/CREPE); pitch via `librosa.yin`.

**Open points and their order are now tracked in the phase sections below** — Phase 1 exit
conditions (seeds, X-ARES, two write-ups), the Phase 2 action question, and Phase 2.5 (long
context + long horizon). See "Sequencing" for the revised order.

Still on-track as designed: continuous codec embeddings (#1), VICReg anti-collapse (#3, holds
— effective rank ~226–241/256), and the closed-loop controllability methodology. Multi-domain
data remains right for the general-audio thesis but is *not* evidence that Phase 1 works — the
gap it was said to close was a measurement artifact. Causal latent prediction (#2) is
implemented and does not collapse, but has not been shown to beat linear extrapolation. Phases 3–4 and the multimodal extension are unreached future work.

---

## Design commitments

These are settled decisions, with the reasoning that led to each:

1. **Input = continuous codec embeddings**, taken *pre-quantization* from a neural codec encoder (EnCodec or DAC). Not mel (we'll keep a mel run only as an A-JEPA-comparable ablation), and explicitly **not LPC / source-filter** — that prior models the vocal tract, not general audio, and breaks on polyphony, percussion, and texture. Continuous (not discrete) embeddings avoid the codec-token-unpredictability problem (cf. LLM-Codec) and come with a trained decoder for the optional render stage.

2. **Objective = causal latent prediction.** Predict future EMA-target embeddings from past context only. Not bidirectional masking (that's A-JEPA / Codec2Vec — a representation learner, not a world model). Not discrete-token cross-entropy (that's AudioLM — forces prediction of local acoustic fluctuations, which is hard and is exactly what latent prediction sidesteps).

3. **Anti-collapse is mandatory, not free.** EMA target encoder + stop-gradient + a VICReg variance/covariance term. This is the key correction over APC: APC could *not* collapse because it regressed to grounded mel frames; we predict a moving EMA target, which can collapse to a constant at zero loss. APC is our architecture-and-horizon template, not our anti-collapse template.

4. **Control = two paths.**
   - *Supervised, domain-general descriptors* defined across all audio: loudness, spectral centroid / envelope (brightness), onset / transient density, and pitch / chroma gated by a voicing flag. The portable kernel of "source-filter" is envelope-vs-fine-structure disentanglement, kept as a soft control axis — not a structural commitment.
   - *Learned latent actions*: an inverse model with a small VQ bottleneck (LAPO / Genie-style), placed mid-stack, for everything descriptors can't express (polyphony, texture, percussion).

5. **Source-filter / DDSP control is quarantined** to a later speech-or-monophonic-instrument variant. Great inductive bias when the signal genuinely is one excitation through one resonator; a liability for the general model.

6. **Decoder is decoupled and optional** — only for generative / controllable-stream use, via the codec's own (frozen) decoder.

---

## Phase 0 — Scaffolding & baselines

- **APC baseline.** Reimplement APC (unidirectional LSTM + residual + time-shift `n`, L1 on the actual frame). ~half a day from the public repos; gives a known-good causal-prediction reference.
- **A-JEPA-style mel baseline** for X-ARES comparability.
- **Codec frontend.** Pick EnCodec or DAC; extract continuous pre-quantizer encoder embeddings; fix the frame rate. Cache embeddings offline (cheap, and lets Phase 1 iterate fast).
- **Data.** AudioSet (general) + a music set (FMA / MTG-Jamendo) + environmental held out for eval (ESC-50 / UrbanSound8K). Deliberately multi-domain from the start to keep the model general-purpose.

## Phase 1 — Causal JEPA pretraining (no control)

The core of the project. Validate this fully before adding anything.

- **Frame encoder** `f_θ`: causal transformer (or conformer) over the codec-embedding sequence → `z_{1..T}`.
- **Target encoder** `f_θ̄`: EMA copy, stop-grad, produces prediction targets.
- **Causal predictor** `g_φ`: causal-masked transformer; from `z_{≤t}` predict EMA-target latents `z_{t+1 .. t+k}`.
- **Loss**: smooth-L1 / cosine in embedding space **+ VICReg variance + covariance**.
- **Horizon**: sweep the time-shift / `k`. APC found a sweet spot around `n=3` on mel; codec embeddings are temporally smoother, so expect a larger optimum. Predict multiple future offsets jointly (cf. Multi-Target APC) to discourage trivial local-smoothness solutions.
- **Validate**: X-ARES linear probe vs A-JEPA / data2vec / wav2vec2; forward latent-prediction error vs a naive persistence baseline across horizons.
- **Watch**: representation variance / effective rank as a live collapse monitor.

### Phase 1 exit conditions (revised 2026-09-01)

Phase 1 is *not* closed. Three things gate it, in this order:

1. ~~Error bars~~ **PASSED (2026-09-01).** Five seeds of the multi-domain config, scored
   against one shared AR(4) floor: codec cos gain **+0.022/+0.027/+0.026/+0.028 ± 0.001**
   and latent skill **+7.7/+12.0/+10.9/+9.3% ± ≤0.3%** at k=1/2/4/8. Weakest horizon at
   mean−1σ is **+0.021**, with zero about 20σ away. The margin is reproducible, not seed
   noise. Read it in proportion, though: AR(4) beats persistence by ~0.09–0.10 here and the
   model beats AR(4) by ~0.025, so the edge over the honest floor is a quarter of the
   floor's edge over the naive one — and it is measured only at 13–107 ms, inside the
   training window.
2. **X-ARES, actually run** *(now the binding constraint on Phase 1)*. The original gate says "competitive on X-ARES". It was never
   run — a homemade ESC-50 std-pooling probe stood in, and that probe is now known to
   measure *readout* rather than information (routing the latent through the model's own
   grounding head before pooling recovers 53.8% vs the codec's 55.0%, from 48.6% raw). So
   criterion (b) is unevaluated, not failed. Run it once properly, or retire the criterion
   explicitly — do not let it lapse quietly.
3. **Two write-ups of experiments already done** (see `RESULTS.md`): the recon-probe result
   above, which replaces "the probe metric is broken" with the provable "the information is
   present but not linearly readable under std-pooling"; and the corrected smoothness claim.
   The measured version of the latter is the more interesting one — an *untrained* causal
   transformer already sits at 0.579 lag-1 autocorrelation vs the codec's 0.291, and an
   untrained LSTM at 0.908 vs trained APC's 0.270. So causal attention over history is
   inherently smoothing; the JEPA objective *preserves* that smoothness where APC's grounded
   objective destroys it. "The causal objective creates smoothness" was never supported.

## Phase 2 — Control conditioning (rebuilt around exogenous interventions)

- **2a — Supervised descriptors.** Extract frame-aligned loudness, spectral centroid, onset density, chroma/pitch (+voicing). Inject via FiLM or cross-attention into `g_φ`. Condition on the *delta* to apply, so control is learned as transition modulation rather than absolute state.
- **2b — Learned latent actions.** Inverse model `q(a_t | z_t, z_{t+1})` with a small VQ bottleneck, inserted mid-stack (VQ-APC found mid-stack insertion best; expect the pretext loss to *worsen* while representation/control quality improves — so don't use prediction loss for model selection). Drop the inverse model at inference; drive with chosen codes.
- **Risk**: latent actions can shortcut / leak. Mitigate with commitment loss, a deliberately small codebook, and entropy/KL regularization on code usage.

### The action question — DECIDED (2026-09-01): exogenous interventions

The descriptors above are **not actions**. In V-JEPA-2-AC the action is *exogenous* — a robot
joint command — and the model must learn the world's response. A descriptor describes the very
observation being predicted, so commanding "loudness +2σ" partially specifies the target
rather than intervening on anything. That is conditional generation, which is why the
controllability eval was only ever measuring conditioning fidelity, and why 2b's codebook
collapsed onto loudness: with no true latent action to recover, a VQ bottleneck can only
quantize the largest innovation, and that is energy.

**Decision: branch (a).** Train on audio with *known applied interventions*, motivated by
robot audition — the microphone is occluded, or the robot moves into a different room.

**What this is, precisely.** These are **observation-model** interventions: covering a
microphone changes how the scene is sensed, not the scene itself. That is the exact audio
analogue of camera pose in V-JEPA-2-AC, and for a mobile robot ego-motion *is* the dominant
agent-controlled cause of audio change — moving alters the source→mic transfer function. So
the defensible claim is **action-conditioned audio prediction**, not "predicts how the world
evolves under its actions". State it that way rather than have a reviewer say it.

#### The design decision that separates a world model from a style transfer

**Interventions are timed events, not per-clip labels.** A uniformly reverberant clip teaches
a static transform. Reverb switching on at t = 2 s because the robot crossed a doorway forces
the model to predict a *transition* — that is action-conditioned dynamics, and it is the only
version that earns the name. It also lands directly on the anticipation eval that already
exists: told "reverb increases now", does the prediction move before the evidence arrives?

#### The three axes

All cheap to synthesise from the cached audio, and all *visually inferable*, which matters for
the multimodal extension below.

| axis | robotics story | action parameter |
|---|---|---|
| **Gain** | microphone covered; distance change | Δ dB |
| **Spectral tilt / lowpass** | microphone occluded, muffled | Δ cutoff (octaves) or tilt dB/oct |
| **Reverb** | moved to another room | Δ RT60, or room parameters directly |

Reverb via `pyroomacoustics` (image-source method) is the best value: RIRs are generated from
room dimensions and absorption, so **the action vector is literally the room parameters** —
continuous, sweepable, and it allows holding out unseen rooms to test generalisation in
*action* space rather than only in audio.

Tier 2, once the above works: source distance/azimuth (needs spatial rendering), and an added
interferer (another agent, a machine starting up).

#### Two rules, both learned expensively in Phase 2a

1. **Post-normalise level after filtering and reverb.** Lowpass removes energy and reverb
   changes level, so without normalisation all three axes correlate with loudness — and the
   loudness axis would be rediscovered for the fourth time.
2. **Condition on the action *delta*, not the state.** Zero everywhere, non-zero at the
   transition. This reuses the existing delta-FiLM machinery unchanged and keeps the semantics
   "an action occurred" rather than "the observation currently looks like this", which is the
   category error that started all of this.

#### Paired data dissolves the recoverability gate

Applied interventions give the **same clip clean and intervened**. Control is then measured by
comparing the predicted future against the *true intervened audio*, not by re-extracting a
descriptor from rendered audio.

That retires the criterion which has governed Phase 2 so far — codec-recoverability of the
descriptor (harmonic_ratio R²≈0.34 works, onset ≈0.07 dead). It matters immediately for
reverb: RT60 is a windowed, not per-frame, quantity — the same property that ruled out the
tempogram — so under the old eval it would have looked undiagnosable. Under paired ground
truth it is fine.

It also supplies the honest 2b re-test: given (clean, intervened) pairs, can the inverse model
recover the applied action? Identifiability is now by construction, so a codebook that still
collapses onto energy would be a real finding rather than a foregone one.

#### Pre-flight results (2026-09-01, 220 ESC-50 clips)

Built and measured before training anything — the discipline that the transient thread cost
us by skipping. `src/tajepa/interventions.py`, `scripts/make_interventions.py`.

**Decorrelation holds.** |corr| of each commanded axis with the measured level change:
gain **0.750** (it is the level axis), tilt **0.006**, reverb **0.029**. Sampled axes are
mutually independent (|corr| 0.08–0.13). Rule 2 works by construction, so the loudness
collapse that killed three Phase 2a attempts cannot recur here.

**Identifiability of the applied action** from the (clean, intervened) codec pair, held out:

| axis | linear | MLP |
|---|---|---|
| gain | **+0.61** | — |
| tilt | +0.25 | **+0.30** |
| reverb (RT60) | −0.62 | **+0.17** |

Reverb repeats the onset lesson exactly: linearly invisible, nonlinearly recoverable. Do not
read a linear null as absence.

**Reverb is strongly content-dependent, and that changed the design.** Splitting the reverb
clips by how transient the *clean* audio is: R² **+0.216** on transient-rich clips versus
**−2.017** on stationary ones. A room is heard through its response to transients; on rain or
wind a reverb change is close to physically invisible. Commanding an action with no observable
consequence is precisely what teaches a predictor to ignore that axis — it is how the Phase 2a
onset dial died — so `sample_spec` now **gates reverb on a transient score** and simply does
not fire it on flat content. Verified: 0/60 firings on stationary noise, 37/60 on impulsive.

*Caveat:* n=220 with a 384-feature probe is under-powered; the ordering is trustworthy, the
absolute values are not. Re-measure on the full training set.

#### First run: the action must LEAD its effect (2026-09-01)

The first trained intervention model **ignored the action**: `action_gain` +0.1–0.4% overall
and 0.0% on tilt and reverb, with `err_without` indistinguishable from `err_with`. The
dead-dial outcome, reproduced despite the redesign.

Not a bug — alignment was checked first and is correct (the effect begins at codec frame 226
while the action marks 225). The cause is structural: the intervention is a **gradual
crossfade (median 8.8 frames) whose audio changes over the same frames as the action**, so a
causal model has already observed the ramp beginning and can extrapolate. Only the ramp's
first frame is unforecastable without the action — 0.4% of a 256-frame window.

This is the persistence mistake in another place. There the baseline already contained the
answer; here the observation already contains the action. Both comparisons looked meaningful
and measured nothing.

**So the design gains one more rule, alongside the timed-event and level-normalisation ones:
the action is commanded `lead` frames before its acoustic effect begins, and the conditioning
window is shifted to match** (`[t−L, t+o−L)`). At the commanded moment nothing is observable,
so conditioning is the only route to anticipation. This is also the honest robotics analogue —
a robot issues a motion command and the acoustic consequence follows; it does not learn of its
own action by hearing it. Note a naive lead *without* the matching window shift makes matters
worse: the action passes out of the conditioning window before the frame it explains.

Secondary levers, both cheap and now available: weighting the loss on event frames, so a
zero-init FiLM receives gradient in proportion to the information rather than 3% of it; and
longer horizons, which couples this to Phase 2.5.

#### Risk: learning to detect the DSP

The failure mode is that the model learns to recognise the applied processing rather than
anything acoustic. Mitigate with real RIRs alongside synthetic ones, randomised intervention
timing and magnitude, and — the load-bearing one — evaluation on **held-out action
parameters**, interpolating and extrapolating in action space rather than only over held-out
audio. A model that works only at the RT60 values it trained on has memorised a filter bank.

#### What carries over

The `ControllableJEPA` FiLM path (action vector in place of descriptor delta), the codec
frontend, and the closed-loop render machinery all survive. The existing descriptor work
becomes the *supervised-descriptor arm* rather than being discarded. What changes is the eval,
which is designed in from the start this time rather than retrofitted.

### Controllability rigour (fold into the intervention eval)

Raw effect sizes in σ (the published matrices are column-normalised, which hides magnitude),
bootstrap CIs, and a percentile rather than clip-mean readout — a mean over 256 frames is
structurally blind to sparse spiky descriptors like onset, which is an alternative
explanation for the "dead" transient dials that has never been separated from the render-path
one.

Note that paired ground truth supersedes most of this for the *intervention* axes — effect
sizes come from comparing against the true intervened audio rather than from a normalised
matrix. What remains necessary is the retrospective check on the existing descriptor results.

**One piece of this is independent and should be done now, not later: the round-trip null.**
Encode true codec frames → decode → re-extract descriptors, with no model involved. It tests
a claim already standing in `RESULTS.md`, that transients are *render-limited* and a
nonlinear latent→codec decoder would fix them. If transients do not survive a **clean** codec
round trip, that diagnosis is wrong and the proposed fix is aimed at the wrong layer. Cheap,
needs no training, and either protects or kills a documented conclusion.

## Phase 2.5 — Long context and long horizon (decided; next phase)

**Replace absolute sinusoidal positional encodings with RoPE or ALiBi.** The encoder
currently adds absolute sinusoidal PE, so the model is only valid below its training
`--window` (256 frames = 3.41 s at 75 Hz). `sinusoidal_pe` is closed-form and returns values
at any index, so over-length inference fails *silently*: past the window the representation
is out of distribution and forward prediction drops below persistence. This inverted a
headline result once (in-window +6% vs AR(4); scored to 512 frames, −15%).

`windowed_predict` is the stopgap, and it has a measured cost: each window boundary leaves a
~20% error spike over 2–3 frames. That is not a history shortage — widening the overlap
barely moves it (stride 128→32: 21%→19%) — but a discontinuity in the target latent
trajectory where two windows meet (`|z[t+1]-z[t]|` is 1.7× its local value at the seam).
Overlap cannot remove a seam; only a single pass over the whole clip can.

RoPE and ALiBi both extrapolate beyond the trained length, which removes the windowing, the
seam, and the silent-failure mode together.

**The horizon sweep belongs here, not as a separate task — and it is blocked by this one.**
The open question the project most needs to answer is whether the model has learned
*dynamics* or merely local continuation: everything measured so far is k=1–8, i.e. 13–107 ms,
a range where "the near future looks like the recent past" explains most of the signal. The
test is a sweep to k=25/75/150 (0.33/1/2 s), far enough out that linear extrapolation should
fail. That cannot be done at the current context: a k=150 pair needs both `t` and `t+150`
inside a 256-frame window, so only the first 106 frames can host one — 41% of the window, and
none of them with much history. k=150 is effectively untrainable today; k=75 is marginal
(71%); k=25 is fine. **A world model needs context ≫ horizon.** So the positional change is a
precondition for the sweep, and the sweep is what justifies the positional change.

Scope this phase as one set of runs:

- Swap absolute sinusoidal PE for RoPE or ALiBi in `CausalTransformer`.
- Train at a longer context (≥1024 frames ≈ 13.7 s) with offsets extended to
  `1 2 4 8 25 75 150`.
- Report skill vs `LinearAR` **as a function of horizon**, with the seeds discipline from
  Phase 1. The interesting result is the horizon at which the AR floor breaks and the model
  does not — if there is no such horizon, "world model" is not earned and the honest framing
  is the Phase 2 branch (b) rename.
- Confirm the windowing seam disappears (it should: one pass, no seam).

This should land before Phase 3: rollout stability over long horizons is not meaningful while
the encoder cannot represent long sequences in the first place.

## Phase 3 — Rollout stability

- Multi-step latent rollout (feed predictions back).
- Scheduled sampling / teacher-forcing anneal to fight exposure bias (the standard AudioLM-family failure under teacher forcing).
- Metric: prediction error vs horizon should degrade gracefully, not blow up. This is where latent world models usually break — budget real time here.

## Phase 4 — Optional decoder / rendering

- Decoupled JEPA-latent → codec-embedding → frozen codec decoder (or a light learned projection into the codec's continuous space).
- Only for generative / controllable-stream use; keep it out of the JEPA core.
- Streaming tie-in: causal + low-frame-rate codec makes real-time generation feasible (Icecast / Liquidsoap).

---

## Evaluation (cross-cutting)

- **Representation quality** — X-ARES linear probe across speech / music / environmental, vs A-JEPA, data2vec, wav2vec2.
- **Predictive quality** — latent MSE vs persistence and vs APC, across horizons.
- **Controllability (closed loop)** — perturb `a_t` (transpose, tempo, brightness, instrument on/off), render, re-run the MIR extractor, and check (a) the intended change happened and (b) unintended attributes stayed fixed (disentanglement).
- **Anti-collapse diagnostics** — embedding variance, effective rank, codebook usage / perplexity.

## Risk register

| Risk | Mitigation |
|---|---|
| Representation collapse (causal + EMA moving target) | EMA + stop-grad + VICReg; live variance/rank monitoring |
| Rollout divergence over long horizons | Scheduled sampling; multi-step training; Phase 3 time budget |
| Codec-token unpredictability | Use continuous pre-quantizer embeddings, not discrete tokens |
| Control shortcutting / leakage | Mid-stack VQ bottleneck; delta-conditioning; code entropy reg |
| Over-specialization to speech | Multi-domain data from Phase 0; source-filter kept to a separate variant |

## Multimodal extension (audio-visual world model)

The single-modality plan above is Phase 1–4. The larger thesis — and the stronger novelty story — is that audio prediction is one sense of an embodied, multisensory world model. Animals don't model sound in isolation; the bicycle-and-headphones intuition is that audio carries action-relevant state (approach, occlusion, off-camera events) that vision misses. The audio JEPA is best conceived as the audio branch of an audio-visual JEPA that shares a predicted latent space with vision.

**Where the frontier actually is (as of mid-2026).** V-JEPA 2 / V-JEPA 2-AC is the embodied template — action-conditioned future-latent prediction, planning via MPC — but vision-only. The audio-visual world model now exists as exactly one entry, AVWM / AV-CDiT (Wang et al., *Audio-Visual World Models: Grounding Multisensory Imagination for Embodied Agents*, [arXiv:2512.00883](https://arxiv.org/abs/2512.00883)), and it is **generative diffusion, not JEPA**: it denoises future visual + audio latents with a DDPM objective and decodes back to pixels and audio. Its soundscape is near-trivial (a single stationary telephone ringtone in synthetic SoundSpaces scenes, four discrete nav actions), it is synthetic-only, and audio is a second-class citizen needing architectural protection from the vision-pretrained backbone. No code or dataset is released. So the JEPA-flavoured, real-audio, balanced-modality version is open.

**The extension (a natural second paper).** Keep the audio branch exactly as planned (causal encoder, EMA target, VICReg, codec embeddings), and add a vision branch plus a cross-modal predictor:

- Two modality-specific encoders (audio: codec-embedding encoder from Phase 1; vision: a ViT or a frozen V-JEPA 2 encoder). Do **not** force a single shared encoder — the shared *space* should emerge from the cross-prediction objective, not from shared tokenization, because audio and video have very different time scales and structure.
- Cross-modal prediction objective: predict video latents from audio context and audio latents from video context, in embedding space, against EMA targets. This is the JEPA analogue of the proven AV-SSL finding (XDC, AVID) that cross-modal prediction is a richer pretext than within-modality — done predictively/causally rather than contrastively, which is the gap.
- Action conditioning stays as in Phase 2; for embodied use, fold in a reward token as AVWM does.

**Lessons to port directly from AV-CDiT (even though it's diffusion):**
- *Modality dominance is real.* A vision-pretrained backbone will swamp audio. Mitigate with per-modality "experts" (separate FFN/projection paths) and a staged schedule: train/adapt vision, then an audio-only stage with shared and visual components frozen, then joint. Budget for this — it's the main reason their model needed three stages.
- *Skip-step Δt prediction* (predict variable horizons, not just next-frame) — the same device as the APC time-shift already in Phase 1. Reuse one horizon-sampling scheme across both the unimodal and multimodal objectives.
- *Reward token as auxiliary output* if targeting planning/navigation.

**The Phase 2 action space is chosen to be visually inferable.** Room geometry predicts
reverb; an occluding hand predicts gain and spectral tilt; approaching a doorway predicts the
transition. So the audio branch's actions are precisely the quantities a vision branch could
later *supply* — which is the cross-modal prediction that makes an AV-JEPA interesting, and
exactly what AVWM lacked with a single stationary ringtone. Honest sequencing, though: all of
Phase 2 is buildable and validatable audio-only, and the AV step still needs synchronised
audio-visual data with ego-motion, which remains the field's blocker (below). Do not tailor
the action space to a specific robot until the audio-only version demonstrably works.

**Data gap = opportunity.** The AVWM authors concede the blocker is the absence of real-world data with both precise action labels and tightly synchronized audio-visual streams. Options: (a) start on their synthetic regime conceptually but with richer audio (multiple/moving sources, music, environmental) to stress the audio branch; (b) use passive in-the-wild AV (no actions) for the cross-modal pretraining objective, which needs no action labels, and reserve action conditioning for a smaller labelled set — mirroring V-JEPA 2's "internet video pretrain, small action-data finetune" split. A real synchronized AV-with-actions dataset would itself be a contribution.

**Sequencing.** This is strictly after the unimodal Phase 1 validates. The cross-modal objective is meaningless if the audio branch hasn't been shown to predict-and-not-collapse on its own first.

## Novelty framing (for paper / proposal)

*Unimodal:* A-JEPA established masked latent prediction for audio; APC established causal future prediction on spectral features. The core contribution unifies them — **causal latent prediction over codec embeddings, plus action conditioning** — into a controllable audio world model. The precise combination (causal, embedding-space, EMA-target/no-decoder, action-conditioned, general-purpose) was not found in the survey; nearest neighbours are APC/CPC (causal latent, but mel and not EMA-JEPA), the AudioLM family (codec tokens + causal, but discrete generative CE), Codec2Vec/MuQ (codec + JEPA-style, but bidirectional), and AudioMNTP (continuous tokens + future prediction, but diffusion-loss generative).

*Multimodal (the stronger story):* the audio-visual world model exists today only in generative-diffusion form (AVWM/AV-CDiT) on a near-toy soundscape. A causal, embedding-space, EMA-target audio-visual JEPA with rich real audio — the predictive cousin of AVWM and the audio extension of V-JEPA 2 — is both differentiated and explicitly named-but-undone in that line's future work. This is where the contribution is largest and the prior art thinnest.

## Stack

PyTorch Lightning · EnCodec/DAC (HF) · madmom / librosa / CREPE for descriptors · AudioSet + FMA/MTG-Jamendo + ESC-50/UrbanSound. Start small — the A-JEPA efficiency result (competitive on <1/5 the data) plus cheap cached codec embeddings means Phase 1 doesn't need large compute to be informative.

## Sequencing

Phase 0: days. Phase 1: the real work, weeks — gate everything on validating it. Phase 2: weeks. Phase 3: the hard part. Phase 4: optional. The multimodal extension is a separate, later effort that depends entirely on Phase 1 succeeding.

**Revised order (2026-09-01), after the evaluation corrections:**

1. **Phase 1 exit conditions** — seeds (in flight), then X-ARES or an explicit retirement of
   that criterion, plus the two write-ups. Cheap, and everything else is uninterpretable
   until the error bars land.
2. ~~The action decision~~ **DECIDED**: exogenous interventions (gain / spectral tilt /
   reverb), applied as timed events. Phase 2 is rebuilt around it — see above. The first
   concrete task is the intervention data pipeline, which is independent of the seeds and
   of Phase 2.5 and can start immediately.
3. **The round-trip null** — independent of both, needs no training, and tests a conclusion
   already in `RESULTS.md`.
4. **Phase 2.5: long context + long horizon** — the one that decides whether "world model"
   is earned.
5. **Phase 2 rebuilt or relabelled**, per (2), with controllability rigour designed in.
6. Phase 3 onward.

The original gate ("beats persistence and is competitive on X-ARES") has been superseded on
its first clause — persistence is not a bar; the floor is a closed-form linear AR — and its
second clause was never actually evaluated. Don't start cross-modal work until both are
settled on evidence.

---

## References

Sources referenced above, with arXiv links (canonical links for the few without an arXiv entry).

**Predictive coding & self-supervised representation learning**
- APC — [An Unsupervised Autoregressive Model for Speech Representation Learning](https://arxiv.org/abs/1904.03240)
- Multi-Target APC — [Improved Speech Representations with Multi-Target Autoregressive Predictive Coding](https://arxiv.org/abs/2004.05274)
- VQ-APC — [Vector-Quantized Autoregressive Predictive Coding](https://arxiv.org/abs/2005.08392)
- CPC — [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- data2vec — [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555)
- wav2vec 2.0 — [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- VICReg — [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)

**JEPA, codec-based SSL & world models**
- A-JEPA — [A-JEPA: Joint-Embedding Predictive Architecture Can Listen](https://arxiv.org/abs/2311.15830)
- Codec2Vec — [Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs](https://arxiv.org/abs/2511.16639)
- MuQ — [MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization](https://arxiv.org/abs/2501.01108)
- V-JEPA 2 / V-JEPA 2-AC — [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- Genie — [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)
- LAPO — [Learning to Act without Actions](https://arxiv.org/abs/2312.10812)
- AVWM / AV-CDiT (Wang et al.) — [Audio-Visual World Models: Grounding Multisensory Imagination for Embodied Agents](https://arxiv.org/abs/2512.00883)

**Neural codecs & generative audio**
- EnCodec — [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
- DAC — [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546)
- AudioLM — [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143)
- LLM-Codec — [UniAudio 1.5: Large Language Model-driven Audio Codec is A Few-shot Audio Task Learner](https://arxiv.org/abs/2406.10056)
- AudioMNTP — [Generative Audio Language Modeling with Continuous-valued Tokens and Masked Next-Token Prediction](https://arxiv.org/abs/2507.09834)

**Audio-visual self-supervised learning**
- XDC — [Self-Supervised Learning by Cross-Modal Audio-Video Clustering](https://arxiv.org/abs/1911.12667)
- AVID — [Audio-Visual Instance Discrimination with Cross-Modal Agreement](https://arxiv.org/abs/2004.12943)

**Control conditioning & descriptors**
- FiLM — [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- DDSP — [DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643)
- CREPE — [CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182)
- madmom — [madmom: a new Python Audio and Music Signal Processing Library](https://arxiv.org/abs/1605.07008)
- librosa — [librosa: Audio and Music Signal Analysis in Python](https://librosa.org/) (SciPy 2015; no arXiv)

**Benchmarks & datasets**
- X-ARES — [X-ARES: A Comprehensive Framework for Assessing Audio Encoder Performance](https://arxiv.org/abs/2505.16369)
- FSD50K — [FSD50K: An Open Dataset of Human-Labeled Sound Events](https://arxiv.org/abs/2010.00475)
- FMA — [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840)
- AudioSet — [Audio Set: An Ontology and Human-Labeled Dataset for Audio Events](https://research.google.com/audioset/) (ICASSP 2017; no arXiv)
- ESC-50 — [ESC: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50) (ACM MM 2015; no arXiv)
- UrbanSound8K — [A Dataset and Taxonomy for Urban Sound Research](https://urbansounddataset.weebly.com/urbansound8k.html) (ACM MM 2014; no arXiv)
- MTG-Jamendo — [The MTG-Jamendo Dataset for Automatic Music Tagging](https://mtg.github.io/mtg-jamendo-dataset/) (ICML 2019 workshop; no arXiv)
