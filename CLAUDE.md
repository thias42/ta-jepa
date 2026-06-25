# CLAUDE.md

## What this project is

A causal, action-conditioned **latent world model for general audio** (music, environmental
sound, speech — *not* speech-only). It predicts *future* audio representations in embedding
space and steers that prediction with control signals: the audio analogue of an
action-conditioned V-JEPA, not a static representation learner.

The full design rationale, phase plan, evaluation, and novelty framing live in
`docs/temporal-audio-jepa-plan.md`. **Read it before making architectural decisions** — this
file is the quick-reference; the plan is the source of truth.

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

## Environment & commands

Python 3.11 in the **conda env `ta-jepa`** (the pyenv 3.11.4 build is broken — no `ssl`
module — do not use it). Interpreter:
`/Users/matthias/miniconda3/envs/ta-jepa/bin/python`. The package is installed editable
(`pip install -e .`). Audio I/O uses `soundfile`, NOT `torchaudio.load` (recent torchaudio
delegates decoding to TorchCodec/FFmpeg, which we deliberately avoid — see `data/io.py`).

```bash
P=/Users/matthias/miniconda3/envs/ta-jepa/bin/python
$P -m pytest -q                                   # fast tests (codec test gated behind env var)
TAJEPA_RUN_CODEC_TESTS=1 $P -m pytest -q           # include EnCodec download/shape test

# End-to-end Phase 0 pipeline (synthetic data -> codec cache -> APC):
$P scripts/make_synthetic_data.py --per-domain 4
$P scripts/build_manifest.py --root data/synthetic/music --domain music \
    --root data/synthetic/environmental --domain environmental \
    --root data/synthetic/speech --domain speech --out data/manifests/synthetic.jsonl
$P scripts/extract_embeddings.py --manifest data/manifests/synthetic.jsonl \
    --cache data/cache/encodec_24khz/synthetic --device cpu
$P scripts/train_apc.py --cache data/cache/encodec_24khz/synthetic --offsets 1 3

# Real data — ESC-50 (environmental, held-out eval; ~616 MB download):
$P scripts/prepare_esc50.py                        # download + extract + manifest
$P scripts/extract_embeddings.py --manifest data/manifests/esc50.jsonl \
    --cache data/cache/encodec_24khz/esc50 --device cpu
```

## Datasets

- **ESC-50** is the first real dataset wired in (smallest of the plan's sets). `prepare_esc50.py`
  → `data/manifests/esc50.jsonl` with `label` (class) + `fold` (official 5-fold CV) on every
  entry; folds map 1-3→train, 4→val, 5→test but `fold` is preserved for proper CV. Held out
  for environmental eval — do NOT pretrain on it. `ManifestEmbeddingDataset` joins cached
  features to labels for the probe.
- Pretraining sets (AudioSet / FMA / MTG-Jamendo) are not yet wired. Large `data/` artifacts
  (downloads, extracted audio, caches) are gitignored (anchored `/data/`).

## Repo layout (Phase 0)

- `src/tajepa/codec/` — frozen codec frontend (`EncodecFrontend`, registry) + offline
  embedding caching (`extract.py`). **Continuous pre-quantizer embeddings only.**
- `src/tajepa/features/mel.py` — log-mel frontend for the A-JEPA-comparable baseline.
- `src/tajepa/models/apc.py` — APC baseline + `persistence_l1` (the bar Phase 1 must beat).
- `src/tajepa/diagnostics.py` — `feature_std` / `effective_rank` collapse monitors, wired
  into training now so the path carries into Phase 1.
- `src/tajepa/data/` — manifests (JSONL), audio + cached-embedding datasets, `io.py`.
- `scripts/` — runnable CLIs; `configs/` — YAML for codec / APC / mel.

## Working conventions

- Cache codec embeddings offline (`extract_embeddings.py`) so the model side iterates fast.
- Keep collapse diagnostics (`diagnostics.py`) wired into every training run.
- Default codec is **EnCodec 24 kHz** (causal-friendly, 75 Hz, HF-native); the frontend is a
  registry so DAC drops in behind the same interface.
- Synthetic data (`make_synthetic_data.py`) is for pipeline smoke-testing ONLY — never for
  evaluation. Real eval is X-ARES on AudioSet/FMA/ESC-50 etc.
