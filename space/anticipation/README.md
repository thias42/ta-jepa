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
(the decoder-free, V-JEPA-style claim). Upload a clip and the demo plots the model's
per-frame prediction error against a persistence baseline under a spectrogram, marks the
**surprise peaks** (least-predictable frames), and reports forecasting skill
(`1 − model/persistence`). Press play and a playhead sweeps both panels in time.

- Code: https://github.com/thias42/ta-jepa
- Model: https://huggingface.co/Maeich/ta-jepa-anticipation

The checkpoint is downloaded at startup from the `MODEL_REPO` Space variable
(default `Maeich/ta-jepa-anticipation`). Drop audio files into an `examples/` folder in this
Space to get one-click example clips.
