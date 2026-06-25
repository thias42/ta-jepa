# CLAUDE.md

## What this project is

A causal, action-conditioned **latent world model for general audio** (music, environmental
sound, speech — *not* speech-only). It predicts *future* audio representations in embedding
space and steers that prediction with control signals: the audio analogue of an
action-conditioned V-JEPA, not a static representation learner.

The full design rationale, phase plan, evaluation, and novelty framing live in
`temporal-audio-jepa-plan.md`. **Read it before making architectural decisions** — this file
is the quick-reference; the plan is the source of truth.

Status (as of this writing): pre-implementation. The repo contains only the plan. There is no
code, no git history, and no environment yet.

## Non-negotiable design invariants

These are settled decisions. Do not silently revisit them; if a change seems to require
violating one, flag it explicitly.

1. **Input = continuous codec embeddings**, taken *pre-quantization* from a neural codec
   encoder (EnCodec or DAC). Not mel (kept only as an A-JEPA-comparable ablation), and
   **not LPC / source-filter** for the general model. Continuous, not discrete — this avoids
   the codec-token-unpredictability problem.
2. **Objective = causal latent prediction.** Predict future EMA-target embeddings from *past
   context only*. Not bidirectional masking (that's a representation learner). Not
   discrete-token cross-entropy (that's the AudioLM family).
3. **Anti-collapse is mandatory.** EMA target encoder + stop-gradient + VICReg
   variance/covariance term. The moving EMA target *can* collapse to a constant at zero loss,
   so collapse diagnostics (embedding variance, effective rank, codebook perplexity) are
   first-class, always-on monitors — not afterthoughts.
4. **Control = two paths.** (a) Supervised domain-general descriptors (loudness, spectral
   centroid/brightness, onset/transient density, pitch/chroma gated by voicing), injected as
   *deltas* via FiLM/cross-attention. (b) Learned latent actions: an inverse model with a
   small VQ bottleneck (LAPO/Genie-style), placed mid-stack.
5. **Source-filter / DDSP control is quarantined** to a later speech-or-monophonic variant.
   Keep it out of the general model.
6. **Decoder is decoupled and optional** — codec's own frozen decoder, for generative use
   only. Never let it leak into the JEPA core.

## Phase structure — and the gate

- **Phase 0** — scaffolding: APC baseline, A-JEPA mel baseline, codec frontend (cache
  pre-quantizer embeddings offline), multi-domain data (AudioSet + FMA/MTG-Jamendo +
  ESC-50/UrbanSound held out). Days.
- **Phase 1** — causal JEPA pretraining, no control. **The real work.** Frame encoder `f_θ`,
  EMA target `f_θ̄`, causal predictor `g_φ`, smooth-L1/cosine + VICReg loss, horizon sweep.
- **Phase 2** — control conditioning (2a descriptors, 2b learned latent actions).
- **Phase 3** — rollout stability (scheduled sampling, multi-step training). The hard part.
- **Phase 4** — optional decoder/rendering, streaming.
- **Multimodal extension** — audio-visual cross-modal JEPA. A separate, later effort.

**THE GATE:** Do not proceed past Phase 1 until the causal audio backbone (a) beats a
persistence baseline on forward latent prediction and (b) is competitive on X-ARES. Do not
start any cross-modal work until that holds. This gating is the central project discipline.

## Gotchas that will bite you

- **Don't use pretext/prediction loss for model selection in Phase 2b.** Adding the VQ
  inverse model is *expected to worsen* prediction loss while representation/control quality
  improves (per VQ-APC). Select on downstream/control metrics, not loss.
- **Horizon is larger than APC's.** APC found `n≈3` on mel; codec embeddings are temporally
  smoother, so expect a bigger optimum. Predict multiple future offsets jointly to discourage
  trivial local-smoothness solutions.
- **Latent actions can shortcut/leak.** Mitigate with commitment loss, a deliberately small
  codebook, and entropy/KL regularization on code usage.
- **APC is the architecture-and-horizon template, NOT the anti-collapse template.** APC
  couldn't collapse (it regressed grounded mel frames); we can. Different problem.
- **Rollout divergence** is where latent world models usually break — budget real time for
  Phase 3, don't shortchange it.

## Stack

PyTorch Lightning · EnCodec/DAC (HF) · madmom / librosa / CREPE for descriptors ·
AudioSet + FMA/MTG-Jamendo + ESC-50/UrbanSound. Start small — cached codec embeddings plus
the A-JEPA efficiency result mean Phase 1 doesn't need large compute to be informative.

## Working conventions

- No build/test/lint commands exist yet. Once an environment is set up, record the canonical
  commands here.
- Cache codec embeddings offline so experiments iterate fast.
- Keep collapse diagnostics wired into every training run from the first commit.
