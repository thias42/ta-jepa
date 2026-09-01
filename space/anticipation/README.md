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
with no training at all. Against it the model is currently **behind on every bundled clip**
(−5% to −20%), and the same holds across ESC-50 and FMA at every horizon tested. So the
demo shows a real, working causal predictor — but not yet one that has learned dynamics a
trivial linear extrapolator hasn't. That gap is the open problem, and the demo reports it
rather than hiding it behind the easier baseline.

The AR reference ships pre-fitted (`assets/ar_latent_fma.pt`), fitted on the checkpoint's own
training data (FMA), so the number is reproducible and doesn't depend on which clips are
bundled.

## Examples

Eight CC0 clips from ESC-50, spanning transient (dog, door knock, glass breaking), tonal
(church bells), rhythmic (clock tick, keyboard typing) and textural (rain, crackling fire)
material — see [`ATTRIBUTION.md`](ATTRIBUTION.md). The rhythmic ones are where a linear
predictor does best and the model's deficit is largest; the transient ones show the clearest
surprise peaks. Drop more files into `examples/` to add your own.

The checkpoint is downloaded at startup from the `MODEL_REPO` Space variable
(default `Maeich/ta-jepa-anticipation`).
