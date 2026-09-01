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
   *(Empirical refinement — see the control modules below / RESULTS.md: spectral descriptors
   are controllable (loudness/brightness/harmonic_ratio work, pitch weak); the plan's
   onset/transient axis is render-limited and does NOT control. Choose axes by
   loudness-decorrelation + codec-recoverability.)*
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

**THE GATE (revised — see the plan's "Sequencing"):** the original wording — beat persistence
and be X-ARES-competitive — is superseded on both clauses. Persistence is not a bar: the floor
is a closed-form **linear AR(4)**, which persistence loses to by a wide margin. And X-ARES was
never actually run; a homemade ESC-50 probe stood in, and that probe measures readout rather
than information. Phase 1 now exits on: (a) skill over `LinearAR` that survives multiple seeds —
**PASSED**: five seeds give +0.022…+0.028 ± 0.001 codec cos gain and +7.7…+12.0% ± ≤0.3%
latent skill, weakest horizon at mean−1σ +0.021 (`evaluate_seeds`); (b) X-ARES run for
real, or that criterion explicitly retired; (c) the two corrections written up. Phase 2 is
**rebuilt around exogenous interventions** (decided 2026-09-01): gain, spectral tilt and
reverb applied to cached audio as *timed events*, so the action is a real intervention rather
than a descriptor of the observation being predicted. Paired clean/intervened data replaces
descriptor re-extraction as the controllability measure, which retires the
codec-recoverability gate. See the plan's Phase 2 for the axes, the two design rules
(post-normalise level; condition on the action delta) and the held-out-action-parameter test;
Phase 2.5 (RoPE/ALiBi + long horizon) is the test of whether "world model" is earned. Do not
start cross-modal work until Phase 1 exits on evidence.

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
- **FMA-small** is wired as the music *pretraining* source. `prepare_fma.py` →
  `data/manifests/fma_small.jsonl` (8000 30 s mp3 tracks, 8 genres, official splits;
  `genre_top` as label). FMA zips are **bzip2** — system `unzip` fails ("PK compat v4.6");
  use Python `zipfile` (the script does) or `7zz`. mp3 decode is via soundfile/libsndfile
  (ffmpeg present as fallback). Embedding extraction is resilient to per-file failures
  (FMA ships a few corrupt mp3s) — failures go to `failures.jsonl` in the cache dir.
- Still unwired: AudioSet / MTG-Jamendo. Large `data/` artifacts (downloads, extracted
  audio, caches) are gitignored (anchored `/data/`).

## Repo layout (Phase 0)

- `src/tajepa/codec/` — frozen codec frontend (`EncodecFrontend`, registry) + offline
  embedding caching (`extract.py`). **Continuous pre-quantizer embeddings only.**
- `src/tajepa/features/mel.py` — log-mel frontend for the A-JEPA-comparable baseline.
- `src/tajepa/models/apc.py` — APC baseline + `persistence_l1` (a sanity floor only —
  the actual bar is `LinearAR`, below).
- `src/tajepa/models/ajepa.py` — A-JEPA mel baseline: masked latent prediction over
  spectrogram patches with an EMA target. Bidirectional, EMA+stop-grad only (NO VICReg —
  that's reserved for the causal JEPA). `train_ajepa.py` owns the EMA target + momentum
  schedule. Probe via `AJEPARepresentation` on cached **log-mel** (`extract_mel.py`).
- `src/tajepa/models/jepa.py` — **Phase 1 core**: causal frame encoder `f_θ` + EMA target
  `f_θ̄` + causal multi-offset predictor `g_φ`; loss = latent smooth-L1 vs stop-grad EMA
  target **+ VICReg variance/covariance** (mandatory anti-collapse, invariant #3).
  `train_jepa.py` owns the EMA target + momentum schedule, records the codec statistics on
  the model (`set_codec_stats`), and logs the forward-prediction L1. Probe via
  `JEPARepresentation` on cached codec embeddings.
- `src/tajepa/models/linear_ar.py` — **`LinearAR`, the forecasting floor.** Closed-form ridge
  AR(p) on the last `p` frames. Persistence (`x[t+k] := x[t]`) is retired as a bar: it is
  cleared trivially on smooth codec embeddings, and AR(4) reproduces (and in latent space
  exceeds) the causal JEPA's entire reported gain over it. Fit on the **training** cache so
  transfer evals stay matched.
- `src/tajepa/data/stats.py` — the one source of truth for codec standardization.
- `src/tajepa/extract.py` — generic feature-cache core; `codec/extract.py` (codec) and
  `extract_mel.py` (log-mel) both delegate to it.
- `src/tajepa/features/descriptors.py` + `src/tajepa/models/control.py` — **Phase 2a control**:
  frame-aligned MIR descriptors as control signals (frontend supports loudness, centroid,
  onset, attack, attack_time, harmonic_ratio, pitch, voicing; the **working default set is
  loudness/centroid/harmonic_ratio/pitch**) + a `ControllableJEPA` whose predictor heads are
  FiLM-conditioned on the descriptor *delta* (control = transition modulation; FiLM zero-init →
  starts unconditioned). Options: `--augment-input` (concat descriptors onto the codec input so
  the encoder can represent them) and `--desc-reg-coef` (ground the control — the predicted
  latent must read as the commanded descriptor). `train_control.py` on a `PairedSequenceDataset`.
  The **closed-loop controllability eval** (`run_controllability.py`) renders predicted latents →
  audio (EnCodec `decode`) → re-extracts MIR → controllability matrix.
  **Result (RESULTS.md): three working dials — loudness, brightness (centroid), harmonic_ratio
  (tonal-vs-noisy); pitch weak; transient axes (onset/attack/attack_time) do NOT render.** Key
  lesson: pick a control axis by *loudness-decorrelation + **codec-recoverability***, which
  predict controllability monotonically (harmonic_ratio R²≈0.34 works, pitch 0.19 weak, onset
  0.07 dead). Transients are the **render-limited frontier** — not a representation problem (the
  codec encodes them nonlinearly; augment puts onset in the latent and it *grounds*), but the
  linear `recon_head` can't draw the transient back into audio. Open lever: a nonlinear
  latent→codec decoder.
- `src/tajepa/models/actions.py` — **Phase 2b learned latent actions** (LAPO/Genie): inverse
  model → small-codebook `VectorQuantizer` → causal `ActionPredictor` (FiLM on the action)
  predicts `z_{t+1}`; drop the inverse model at inference, drive with chosen codes
  (`predict_with_actions`). `codebook_perplexity` monitors usage/leakage; the eval
  (`run_action_eval.py`) scores per-code effect / consistency / separability.
  **Result: the codebook collapsed to a loudness axis** (all 16 codes ≈ loudness levels) — the
  learned actions rediscover the dominant predictable axis, not the transient/texture control
  hoped for. (Don't select on pretext loss — watch perplexity + the eval.)
- `src/tajepa/models/residual.py` — **Phase 2a+2b residual actions**: predictor gets the
  descriptor delta for free (FiLM) so the small VQ codebook is pushed onto the *residual*
  transition (`train_residual.py`, `run_residual_eval.py`, Modal `train_residual`/`residual_eval`).
  **Result: partial — 4/16 codes became onset-dominant (vs 0/16 in pure 2b) but weak and
  inconsistent**; transients stayed render-limited (same wall as 2a).
- `src/tajepa/interventions.py` + `scripts/make_interventions.py` — **Phase 2 exogenous
  actions**: gain / spectral tilt / reverb applied to audio as *timed events*, with the
  per-frame action delta. Level is re-normalised after tilt and reverb so **gain is the only
  axis that changes level** (measured |corr| with Δlevel: gain 0.75, tilt 0.006, reverb 0.03)
  — this is what stops the loudness collapse that killed three Phase 2a attempts. Reverb is
  **gated on a transient score**: a room is heard through its response to transients, and the
  applied RT60 is recoverable at R² +0.22 on transient-rich clips vs −2.0 on stationary ones,
  so firing it on rain or wind would command an action with no observable consequence — how
  the onset dial died. Output is a paired `.npz` cache (clean / intervened / action) read by
  `InterventionPairDataset`; both sides are encoded in one pass, so they are frame-aligned.
- `scripts/train_intervention.py` + `src/tajepa/models/action_conditioning.py` — the
  action-conditioned model. `ControllableJEPA` with the action in place of the descriptor
  delta; conditioning at offset `o` is the action summed over `[t, t+o)` (standard
  convention: `a_t` takes the world from `t` to `t+1`). The encoder does **not** see the
  action — an action is not an observation, and feeding it in would re-create the
  circularity the redesign exists to remove. Prediction loss is logged separately on event
  vs action-free frames, because the aggregate is dominated by the latter and would hide
  whether the action is used at all.
- `src/tajepa/eval/intervention.py` + `scripts/run_counterfactual.py` — **the eval paired
  data buys**: predict the future twice from one context, with and without the applied
  action, both scored against the true intervened future.
  `action_gain = 1 - err(with)/err(without)`; ~0 is a dead dial. Content difficulty cancels.
  Per-axis figures use only single-axis clips so no axis borrows credit from another, and
  are **stratified by observability** (median split on clean-audio codec flux). Reverb is
  the motivating case — it is heard through transients, recoverable at R² +0.22 on
  transient-rich clips vs −2.0 on stationary ones — so the split separates "inaudible here"
  from a dead dial, and flags the reverse failure: an axis working where its effect should
  be inaudible is reading the applied DSP, not the acoustics. **The expected ratio is
  per-axis**: gain is equally audible on both halves, so ~1.0 is correct for it; the test
  bites on rt60.
- Firing rates on FMA (checked): gain 59% / tilt 61% / reverb 58%, only 1% of clips below
  the reverb gate (ESC-50: 10%). Music is consistently moderate in transient content;
  ESC-50 is bimodal — sharp events plus a stationary tail.
- `src/tajepa/diagnostics.py` — `feature_std` / `effective_rank` collapse monitors, wired
  into training now so the path carries into Phase 1.
- `src/tajepa/data/` — manifests (JSONL), audio + cached-embedding datasets, `io.py`.
- `scripts/` — runnable CLIs; `configs/` — YAML for codec / APC / mel.

## Cloud (Modal + R2)

`modal_app.py` runs extraction/training/eval on Modal serverless GPUs with Cloudflare R2
storage (caches + checkpoints as tarballs). We train on cached embeddings (~10–15 GB), not
raw audio, so training is cheap and I/O-light; extraction runs on Modal from public sources
(no audio upload). Each Modal function syncs R2→local and shells out to the existing scripts,
so cloud == local code. Setup + commands: `docs/cloud-modal.md`. Cloud deps: `pip install -e
".[cloud]"` (modal, boto3). Can't be run/tested without the user's Modal auth + R2 creds.

## Working conventions

- Cache codec embeddings offline (`extract_embeddings.py`) so the model side iterates fast.
- Keep collapse diagnostics (`diagnostics.py`) wired into every training run.
- **Never report forecasting skill against persistence alone** — always against `LinearAR`
  (`--ar-order`, default 4). A result that only beats persistence is not a result.
- **Never score past the trained context.** The encoder uses *absolute* sinusoidal
  positional encodings, so the model is only valid below its training `--window` (default
  256 = 3.41 s at 75 Hz). `sinusoidal_pe` returns values for any index, so over-length
  inference fails silently: past the window the forecast drops *below persistence*, which
  once inverted a headline (in-window +6% vs AR(4), scored to 512 frames −15%). Training
  records the window as `context_frames`; eval clamps to it by default; long audio goes
  through `windowed_predict` — a stopgap that leaves a ~20% error spike at each window seam
  (a discontinuity in the recomputed target latents, not a history shortage; more overlap
  does not help). **Decided: move the encoder to RoPE or ALiBi** (plan, Phase 2.5) before
  Phase 3 — rollout stability is not meaningful while the encoder cannot represent long
  sequences at all.
- **The grounding head emits into a recorded space.** `grounding_loss` requires explicit
  `mean`/`std`; training computes them once via `data.stats.codec_stats` and stores them on
  the model, so they travel in the checkpoint. Anything consuming the head's output must use
  `reconstruct_raw` — never re-derive statistics at the point of use. Scoring the head
  against eval-set statistics once flipped a headline forecasting result from +0.08 to −0.07
  and produced a "decoder doesn't transfer" finding that did not exist. Pre-stats
  checkpoints load but warn; pass `--train-cache` / `--train-stats`.
- Default codec is **EnCodec 24 kHz** (causal-friendly, 75 Hz, HF-native); the frontend is a
  registry so DAC drops in behind the same interface.
- Synthetic data (`make_synthetic_data.py`) is for pipeline smoke-testing ONLY — never for
  evaluation. Real eval is X-ARES on AudioSet/FMA/ESC-50 etc.
