# ta-jepa

**Temporally-controlled general-purpose audio JEPA** — a causal, action-conditioned latent
world model for general audio (music, environmental sound, speech). It predicts *future*
audio representations in embedding space and steers that prediction with control signals: the
audio analogue of an action-conditioned V-JEPA, not a static representation learner.

- Design rationale, phases, evaluation, novelty: [`docs/temporal-audio-jepa-plan.md`](docs/temporal-audio-jepa-plan.md)
- Quick-reference invariants & commands for contributors (incl. Claude): [`CLAUDE.md`](CLAUDE.md)

## Status — Phase 0 (scaffolding & baselines)

In progress. What's implemented and verified end-to-end:

- **Codec frontend** — frozen EnCodec, continuous *pre-quantizer* embeddings (75 Hz, dim 128).
- **Offline embedding cache** — `[T, D]` `.npy` per clip + `meta.yaml`.
- **APC baseline** — causal LSTM + residual + multi-offset time-shift, L1 on the actual
  frame; includes a naive persistence baseline (the bar Phase 1 must beat).
- **A-JEPA mel baseline** — masked latent prediction over spectrogram patches with an EMA
  target encoder (bidirectional; X-ARES-comparable). Faithful to I-JEPA/A-JEPA — EMA +
  stop-grad only, no VICReg (that's reserved for our causal JEPA).
- **Log-mel frontend** + offline mel caching.
- **Collapse diagnostics** — feature std / effective rank, wired into training.
- **Data plumbing** — JSONL manifests (with class label / CV fold), audio +
  cached-embedding datasets (incl. a label-joined `ManifestEmbeddingDataset` for probes),
  synthetic data generator for smoke-testing.
- **ESC-50** — environmental eval set: `scripts/prepare_esc50.py` downloads, extracts, and
  builds a manifest (2000 clips, 50 classes, official 5 folds → train/val/test). Held out.
- **FMA-small** — music *pretraining* source: `scripts/prepare_fma.py` extracts and builds a
  manifest (8000 30 s mp3 tracks, 8 genres, official splits; `genre_top` kept as a label).
  Extraction is resilient to FMA's known-corrupt mp3s (failures logged, run continues).

Sanity check on the synthetic set: APC reaches L1 ≈ 1.68 vs persistence ≈ 2.52 at offset 3.

Still open in Phase 0: more pretraining data (AudioSet / MTG-Jamendo), and a full
FMA-pretrained run of the APC / A-JEPA baselines probed on ESC-50 (vs the codec baseline
below) — the trainers and eval are in place; what's left is the compute.

## Setup

Uses a conda env on Python 3.11 (see `CLAUDE.md` for why pyenv 3.11.4 is unusable here).

```bash
conda create -y -n ta-jepa python=3.11
conda run -n ta-jepa pip install -e ".[dev]"
```

## Quickstart (synthetic, runs on CPU)

```bash
P=$(conda run -n ta-jepa which python)
$P scripts/make_synthetic_data.py --per-domain 4
$P scripts/build_manifest.py --root data/synthetic/music --domain music \
    --root data/synthetic/environmental --domain environmental \
    --root data/synthetic/speech --domain speech --out data/manifests/synthetic.jsonl
$P scripts/extract_embeddings.py --manifest data/manifests/synthetic.jsonl \
    --cache data/cache/encodec_24khz/synthetic --device cpu
$P scripts/train_apc.py --cache data/cache/encodec_24khz/synthetic --offsets 1 3
```

## Real data: ESC-50 (environmental eval)

```bash
P=$(conda run -n ta-jepa which python)
$P scripts/prepare_esc50.py                      # download + extract + manifest
$P scripts/extract_embeddings.py \
    --manifest data/manifests/esc50.jsonl \
    --cache data/cache/encodec_24khz/esc50 --device cpu
```

`ManifestEmbeddingDataset(manifest, cache, split="train")` then yields cached features joined
to integer-encoded class labels and CV folds — the input to the X-ARES-style linear probe.

## A-JEPA mel baseline

```bash
P=$(conda run -n ta-jepa which python)
# cache log-mel for the pretraining set (and the probe set), then pretrain:
$P scripts/extract_mel.py --manifest data/manifests/fma_small.jsonl \
    --cache data/cache/logmel/fma_small --config configs/mel_baseline.yaml
$P scripts/train_ajepa.py --cache data/cache/logmel/fma_small \
    --dim 256 --depth 6 --mask-ratio 0.6 --max-steps 20000 --save runs/ajepa.ckpt
# probe it on ESC-50 (cache ESC-50 mel first with extract_mel.py):
$P scripts/run_probe.py --manifest data/manifests/esc50.jsonl \
    --cache data/cache/logmel/esc50 --representation ajepa --ajepa-ckpt runs/ajepa.ckpt
```

## Real data: FMA-small (music pretraining)

```bash
P=$(conda run -n ta-jepa which python)
$P scripts/prepare_fma.py --download             # ~7.5 GB; or place zips in data/downloads/
$P scripts/extract_embeddings.py \
    --manifest data/manifests/fma_small.jsonl \
    --cache data/cache/encodec_24khz/fma_small --device cpu
```

## Tests

```bash
conda run -n ta-jepa pytest -q
# include the EnCodec download/shape test:
TAJEPA_RUN_CODEC_TESTS=1 conda run -n ta-jepa pytest -q
```
