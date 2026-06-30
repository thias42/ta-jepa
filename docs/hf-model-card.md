---
license: mit
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - audio
  - world-model
  - jepa
  - self-supervised
  - causal
  - forecasting
  - encodec
---

# ta-jepa — Phase 1 causal audio JEPA (FMA + FSD50K, grounded)

A **causal, latent world model for general audio**. From past context only it predicts
*future* audio representations in embedding space — the audio analogue of a causal V-JEPA,
not a static representation learner.

- **Input:** EnCodec 24 kHz **continuous pre-quantizer** embeddings (75 Hz, dim 128).
- **Architecture:** causal transformer encoder `f_θ` + EMA target encoder `f_θ̄` +
  causal multi-offset predictor `g_φ`; loss = latent smooth-L1 vs stop-grad EMA target
  **+ VICReg** variance/covariance (anti-collapse). Offsets 1/2/4/8.
- **Training data:** FMA-small (music) + FSD50K (general audio).
- **Result:** on held-out **ESC-50** it beats latent persistence at every horizon and
  matches APC on transfer; no collapse (effective rank ~226/256). The forecasting-vs-horizon
  metric — not the std-pooling linear probe — is the right yardstick for a world model
  (the causal objective makes the latent temporally smooth, which the probe penalizes).
  See the repo's `RESULTS.md`.

This file: `jepa_fma_grounded.ckpt` — a PyTorch Lightning checkpoint (state dict +
hyper-parameters). ~103 MB.

## Usage (Lightning-free load)

```python
import copy, torch
from huggingface_hub import hf_hub_download
from tajepa.models.jepa import JEPA          # pip install "tajepa @ git+https://github.com/thias42/ta-jepa.git"

ck = torch.load(hf_hub_download("Maeich/ta-jepa-anticipation", "jepa_fma_grounded.ckpt"),
                map_location="cpu", weights_only=False)
hp, sd = ck["hyper_parameters"], ck["state_dict"]
jepa = JEPA(in_dim=hp["in_dim"], dim=hp["dim"], enc_depth=hp["enc_depth"],
            pred_depth=hp["pred_depth"], heads=hp["heads"],
            offsets=tuple(hp["offsets"]), dropout=hp.get("dropout", 0.0))
jepa.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("jepa.")}, strict=False)
target = copy.deepcopy(jepa.encoder)         # EMA target encoder (prediction targets)
target.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("target.")})
jepa.eval(); target.eval()
```

## Links

- Code: https://github.com/thias42/ta-jepa
- Interactive demo (Space): https://huggingface.co/spaces/Maeich/ta-jepa-anticipation-demo

## License

MIT.
