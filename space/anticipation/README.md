---
title: ta-jepa Anticipation
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.19.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
models:
  - Maeich/ta-jepa-anticipation
---

# ta-jepa — anticipation

A causal audio **world model** that predicts the near future of a sound *in latent space*
(the decoder-free, V-JEPA-style claim). Pick a clip — or upload one — and the demo plots the
model's per-frame prediction error under a spectrogram, marks the **surprise peaks**
(least-predictable frames), and reports forecasting skill against two references. Press play
and a playhead sweeps both panels in time.

- Code: https://github.com/thias42/ta-jepa
- Model: https://huggingface.co/Maeich/ta-jepa-anticipation

## Two baselines, and why the second one matters

**Persistence** ("the next moment equals this one") is the intuitive reference, and the model
beats it comfortably — typically +13% to +29% on the bundled clips. That number flatters the
model. Codec embeddings are temporally smooth, so persistence is a very easy bar.

**Linear AR(4)** is the honest one: a closed-form least-squares fit on the last four latents,
with no training at all. The model beats it on every bundled clip, but only by **+1% to +16%**
— roughly a tenth of its apparent margin over persistence. That ratio is the point: a model
can look transformative against persistence and merely decent against a baseline that costs
nothing to fit.

The AR reference ships pre-fitted (`assets/ar_latent_fma.pt`), fitted on the checkpoint's own
training data (FMA) inside its training window, so the number is reproducible and doesn't
depend on which clips are bundled.

## The 3.41-second context window

The model was trained on 256-frame sequences (3.41 s at 75 Hz) and uses *absolute* positional
encodings, so it is only valid below that length. Run a longer clip in one pass and the
prediction degrades to **worse than persistence** the moment it crosses 3.41 s — not because
the audio gets harder, but because those positions were never learned. The same frames re-fed
inside a 256-frame window score normally.

The demo therefore runs long audio through overlapping ≤256-frame windows, so every frame you
see has both an in-range position and real history behind it. Lifting the limit properly would
mean a positional scheme that extrapolates (RoPE/ALiBi) or training on longer windows.

## Examples

Eight CC0 clips from ESC-50, spanning transient (dog, door knock, glass breaking), tonal
(church bells), rhythmic (clock tick, keyboard typing) and textural (rain, crackling fire)
material — see [`ATTRIBUTION.md`](ATTRIBUTION.md). The rhythmic ones are where a linear
predictor does best and the model's deficit is largest; the transient ones show the clearest
surprise peaks. Drop more files into `examples/` to add your own.

The checkpoint is downloaded at startup from the `MODEL_REPO` Space variable
(default `Maeich/ta-jepa-anticipation`).
