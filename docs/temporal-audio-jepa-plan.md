# Temporally-Controlled General-Purpose Audio JEPA — Implementation Plan

A causal, action-conditioned latent world model for general audio (music, environmental sound, speech), not a speech-only model. The goal is to predict *future* audio representations in embedding space and to steer that prediction with control signals — i.e. the audio analogue of an action-conditioned V-JEPA, not just a static representation learner.

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

## Phase 2 — Control conditioning

- **2a — Supervised descriptors.** Extract frame-aligned loudness, spectral centroid, onset density, chroma/pitch (+voicing). Inject via FiLM or cross-attention into `g_φ`. Condition on the *delta* to apply, so control is learned as transition modulation rather than absolute state.
- **2b — Learned latent actions.** Inverse model `q(a_t | z_t, z_{t+1})` with a small VQ bottleneck, inserted mid-stack (VQ-APC found mid-stack insertion best; expect the pretext loss to *worsen* while representation/control quality improves — so don't use prediction loss for model selection). Drop the inverse model at inference; drive with chosen codes.
- **Risk**: latent actions can shortcut / leak. Mitigate with commitment loss, a deliberately small codebook, and entropy/KL regularization on code usage.

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

**Where the frontier actually is (as of mid-2026).** V-JEPA 2 / V-JEPA 2-AC is the embodied template — action-conditioned future-latent prediction, planning via MPC — but vision-only. The audio-visual world model now exists as exactly one entry, AVWM / AV-CDiT (Wang et al., arXiv 2512.00883), and it is **generative diffusion, not JEPA**: it denoises future visual + audio latents with a DDPM objective and decodes back to pixels and audio. Its soundscape is near-trivial (a single stationary telephone ringtone in synthetic SoundSpaces scenes, four discrete nav actions), it is synthetic-only, and audio is a second-class citizen needing architectural protection from the vision-pretrained backbone. No code or dataset is released. So the JEPA-flavoured, real-audio, balanced-modality version is open.

**The extension (a natural second paper).** Keep the audio branch exactly as planned (causal encoder, EMA target, VICReg, codec embeddings), and add a vision branch plus a cross-modal predictor:

- Two modality-specific encoders (audio: codec-embedding encoder from Phase 1; vision: a ViT or a frozen V-JEPA 2 encoder). Do **not** force a single shared encoder — the shared *space* should emerge from the cross-prediction objective, not from shared tokenization, because audio and video have very different time scales and structure.
- Cross-modal prediction objective: predict video latents from audio context and audio latents from video context, in embedding space, against EMA targets. This is the JEPA analogue of the proven AV-SSL finding (XDC, AVID) that cross-modal prediction is a richer pretext than within-modality — done predictively/causally rather than contrastively, which is the gap.
- Action conditioning stays as in Phase 2; for embodied use, fold in a reward token as AVWM does.

**Lessons to port directly from AV-CDiT (even though it's diffusion):**
- *Modality dominance is real.* A vision-pretrained backbone will swamp audio. Mitigate with per-modality "experts" (separate FFN/projection paths) and a staged schedule: train/adapt vision, then an audio-only stage with shared and visual components frozen, then joint. Budget for this — it's the main reason their model needed three stages.
- *Skip-step Δt prediction* (predict variable horizons, not just next-frame) — the same device as the APC time-shift already in Phase 1. Reuse one horizon-sampling scheme across both the unimodal and multimodal objectives.
- *Reward token as auxiliary output* if targeting planning/navigation.

**Data gap = opportunity.** The AVWM authors concede the blocker is the absence of real-world data with both precise action labels and tightly synchronized audio-visual streams. Options: (a) start on their synthetic regime conceptually but with richer audio (multiple/moving sources, music, environmental) to stress the audio branch; (b) use passive in-the-wild AV (no actions) for the cross-modal pretraining objective, which needs no action labels, and reserve action conditioning for a smaller labelled set — mirroring V-JEPA 2's "internet video pretrain, small action-data finetune" split. A real synchronized AV-with-actions dataset would itself be a contribution.

**Sequencing.** This is strictly after the unimodal Phase 1 validates. The cross-modal objective is meaningless if the audio branch hasn't been shown to predict-and-not-collapse on its own first.

## Novelty framing (for paper / proposal)

*Unimodal:* A-JEPA established masked latent prediction for audio; APC established causal future prediction on spectral features. The core contribution unifies them — **causal latent prediction over codec embeddings, plus action conditioning** — into a controllable audio world model. The precise combination (causal, embedding-space, EMA-target/no-decoder, action-conditioned, general-purpose) was not found in the survey; nearest neighbours are APC/CPC (causal latent, but mel and not EMA-JEPA), the AudioLM family (codec tokens + causal, but discrete generative CE), Codec2Vec/MuQ (codec + JEPA-style, but bidirectional), and AudioMNTP (continuous tokens + future prediction, but diffusion-loss generative).

*Multimodal (the stronger story):* the audio-visual world model exists today only in generative-diffusion form (AVWM/AV-CDiT) on a near-toy soundscape. A causal, embedding-space, EMA-target audio-visual JEPA with rich real audio — the predictive cousin of AVWM and the audio extension of V-JEPA 2 — is both differentiated and explicitly named-but-undone in that line's future work. This is where the contribution is largest and the prior art thinnest.

## Stack

PyTorch Lightning · EnCodec/DAC (HF) · madmom / librosa / CREPE for descriptors · AudioSet + FMA/MTG-Jamendo + ESC-50/UrbanSound. Start small — the A-JEPA efficiency result (competitive on <1/5 the data) plus cheap cached codec embeddings means Phase 1 doesn't need large compute to be informative.

## Sequencing

Phase 0: days. Phase 1: the real work, weeks — gate everything on validating it. Phase 2: weeks. Phase 3: the hard part. Phase 4: optional. The multimodal extension is a separate, later effort that depends entirely on Phase 1 succeeding. Don't proceed past Phase 1 until the causal audio backbone beats persistence and is competitive on X-ARES — and don't start the cross-modal work until it does.
