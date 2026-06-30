"""Hugging Face Space entrypoint — ta-jepa anticipation demo.

Downloads the Phase 1 causal JEPA checkpoint from a HF model repo and launches the shared
anticipation demo (`tajepa.demo.anticipation`). The ta-jepa package itself is installed from
GitHub via requirements.txt, so this file only loads weights and wires the UI.

Loading is Lightning-free (plain torch.load + JEPA) so the Space doesn't depend on
pytorch-lightning's checkpoint machinery at runtime.

Config via Space variables (Settings → Variables):
  MODEL_REPO  HF model repo holding the checkpoint   (default: Maeich/ta-jepa-anticipation)
  CKPT_FILE   checkpoint filename within that repo    (default: jepa_fma_grounded.ckpt)
"""

from __future__ import annotations

import copy
import os

import torch
from huggingface_hub import hf_hub_download

from tajepa.codec.frontend import build_frontend
from tajepa.config import CodecConfig
from tajepa.demo.anticipation import HEAD_JS, build_anticipation_demo
from tajepa.models.jepa import JEPA

MODEL_REPO = os.environ.get("MODEL_REPO", "Maeich/ta-jepa-anticipation")
CKPT_FILE = os.environ.get("CKPT_FILE", "jepa_fma_grounded.ckpt")


def _sub_state(sd: dict, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in sd.items() if k.startswith(prefix)}


def load_jepa(path: str):
    """Rebuild the JEPA + EMA target from a Lightning checkpoint, weights only."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp, sd = ck["hyper_parameters"], ck["state_dict"]
    jepa = JEPA(in_dim=hp["in_dim"], dim=hp["dim"], enc_depth=hp["enc_depth"],
                pred_depth=hp["pred_depth"], heads=hp["heads"],
                offsets=tuple(hp["offsets"]), dropout=hp.get("dropout", 0.0))
    jepa.load_state_dict(_sub_state(sd, "jepa."), strict=False)   # recon_head may be absent
    target = copy.deepcopy(jepa.encoder)
    target.load_state_dict(_sub_state(sd, "target."))
    return jepa.eval(), target.eval()


ckpt_path = hf_hub_download(MODEL_REPO, CKPT_FILE)
jepa, target = load_jepa(ckpt_path)
codec = build_frontend(CodecConfig(device="cpu"))           # Spaces free tier is CPU
examples = "examples" if os.path.isdir("examples") else None
demo = build_anticipation_demo(jepa, target, codec, max_seconds=12.0, examples=examples)

if __name__ == "__main__":
    demo.launch(head=HEAD_JS, allowed_paths=(["examples"] if examples else None))
