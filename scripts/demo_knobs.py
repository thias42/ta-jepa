"""Web demo — steer a clip's near-future along ta-jepa's working control dials.

Loads a Phase 2a ``ControllableJEPA`` checkpoint (``train_control.py``) and exposes the
*validated* control axes as sliders in a Gradio web UI. Given an input clip we:

  1. encode it to continuous EnCodec embeddings (raw, pre-quantizer),
  2. extract + per-clip-standardize the frame-aligned MIR descriptors,
  3. predict the near future with the **observed** deltas — the model's neutral render,
  4. predict again with each slider **added to its descriptor's delta** — the steered render,
  5. un-standardize the grounded latent (recon head outputs standardized codec embeddings,
     see ``grounding_loss``) and decode it back to audio with EnCodec's own decoder.

The honest comparison is **baseline vs steered**: both pass through the same lossy *linear*
grounding head, so the audible difference is the control effect, not the render loss. The
codec round-trip is shown as a fidelity reference. Loudness/brightness/harmonic_ratio are
the three dials that survived the closed-loop eval; pitch is weak and the transient axes
(onset/attack/attack_time) are render-limited, so they are off the board by default
(see RESULTS.md / ``run_controllability.py``).

    python scripts/demo_knobs.py --ckpt runs/control.ckpt \
        --names loudness centroid harmonic_ratio --examples data/cache/../demo_clips

Requires the demo extra:  pip install -e ".[demo]"   (adds gradio)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import torch

from tajepa.config import CodecConfig, DescriptorConfig, resolve_device
from tajepa.codec.frontend import build_frontend
from tajepa.features.descriptors import DescriptorFrontend
from tajepa.data.io import load_resampled

# Friendly slider labels per descriptor. The "+"/"−" hints describe what bumping the
# standardized delta upward does. Bracketed caveats are honest about the weak/dead axes.
_LABELS = {
    "loudness": "Loudness   (−  softer · louder  +)",
    "centroid": "Brightness   (−  darker · brighter  +)",
    "harmonic_ratio": "Tonal ↔ noisy   (−  noisier · more tonal  +)",
    "pitch": "Pitch   (−  lower · higher  +)   [weak]",
    "onset": "Onset density   [render-limited]",
    "attack": "Attack   [render-limited]",
    "attack_time": "Attack time   [render-limited]",
    "voicing": "Voicing",
}


def _peak_norm(*clips: np.ndarray) -> list[np.ndarray]:
    """Scale a group of clips by their *shared* peak so loudness differences between them
    survive (per-clip normalization would erase the loudness dial) while avoiding clip."""
    peak = max((float(np.abs(c).max()) for c in clips if c.size), default=1.0)
    s = 1.0 / max(peak, 1.0)
    return [c * s for c in clips]


def build_demo(args):
    import gradio as gr

    device = args.device or resolve_device("auto")

    # ControlLightning lives in scripts/ alongside this file (matches run_controllability).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_control import ControlLightning

    lit = ControlLightning.load_from_checkpoint(str(args.ckpt), map_location="cpu")
    model = lit.model.eval().to(device)

    names = list(args.names)
    if len(names) != int(model.cond_dim):
        raise SystemExit(
            f"--names has {len(names)} descriptor(s) {names} but the checkpoint expects "
            f"cond_dim={model.cond_dim}. Pass the same descriptors the model was trained on."
        )
    offset = args.offset if args.offset is not None else min(model.offsets)
    if offset not in model.offsets:
        raise SystemExit(f"--offset {offset} not in model offsets {model.offsets}")

    codec = build_frontend(CodecConfig(device=device))
    desc_fe = DescriptorFrontend(DescriptorConfig(names=tuple(names)))
    sr = int(codec.sample_rate)
    _cache: dict = {}

    @torch.no_grad()
    def _ingest(path: str):
        """Encode + descriptor-extract once per clip; cache the expensive parts."""
        key = (path, args.max_seconds)
        if key in _cache:
            return _cache[key]
        wav = load_resampled(path, sr, mono=True)              # [1, N]
        if args.max_seconds:
            wav = wav[:, : int(args.max_seconds * sr)]
        x = codec.encode(wav.unsqueeze(0).to(device))          # [1, T, Dc] raw
        desc = desc_fe.encode(wav.unsqueeze(0)).to(device)     # [1, T, C]
        t = min(x.shape[1], desc.shape[1])
        x, desc = x[:, :t], desc[:, :t]
        flat = desc.reshape(-1, desc.shape[-1])
        ctrl = (desc - flat.mean(0)) / flat.std(0).clamp_min(1e-4)   # per-clip z-score
        xf = x.reshape(-1, x.shape[-1])
        mu, sd = xf.mean(0), xf.std(0).clamp_min(1e-4)         # un-standardization stats
        roundtrip = codec.decode(x).squeeze().detach().cpu().numpy()  # raw -> audio
        out = (x, ctrl, mu, sd, roundtrip)
        _cache[key] = out
        return out

    @torch.no_grad()
    def _render(x, ctrl, mu, sd, bumps: dict[str, float]) -> np.ndarray:
        base = model.deltas_from(ctrl)
        deltas = {o: base[o].clone() for o in base}
        for name, val in bumps.items():
            deltas[offset][..., names.index(name)] += float(val)
        _, preds = model.predict_with_deltas(x, deltas, desc=ctrl)
        emb = model.reconstruct(preds[offset]) * sd + mu       # un-standardize grounded latent
        return codec.decode(emb).squeeze().detach().cpu().numpy()

    def run(path, *slider_vals):
        if not path:
            return None, None, None
        x, ctrl, mu, sd, roundtrip = _ingest(path)
        bumps = {n: v for n, v in zip(names, slider_vals)}
        baseline = _render(x, ctrl, mu, sd, {n: 0.0 for n in names})
        steered = _render(x, ctrl, mu, sd, bumps)
        roundtrip, baseline, steered = _peak_norm(roundtrip, baseline, steered)
        return (sr, roundtrip), (sr, baseline), (sr, steered)

    with gr.Blocks(title="ta-jepa control knobs") as demo:
        gr.Markdown(
            "# ta-jepa — control knobs\n"
            "A causal audio **world model**: it predicts the near future of a sound and lets "
            "you *steer* that prediction. Load a clip, turn the dials (units are standard "
            "deviations of the steered delta), and render.\n\n"
            "**Listen to *Model (neutral)* vs *Steered*** — both pass through the same linear "
            "render head, so the difference you hear is the control. *Codec round-trip* is the "
            "fidelity ceiling (no model)."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(type="filepath", label="Input clip", sources=["upload", "microphone"])
                sliders = [
                    gr.Slider(-3.0, 3.0, value=0.0, step=0.25, label=_LABELS.get(n, n))
                    for n in names
                ]
                btn = gr.Button("Render", variant="primary")
            with gr.Column():
                out_rt = gr.Audio(label="Codec round-trip (no model — fidelity reference)")
                out_base = gr.Audio(label="Model (neutral prediction)")
                out_steer = gr.Audio(label="Steered (your dials applied)")
        if args.examples:
            ex = sorted(p for p in Path(args.examples).glob("*") if p.suffix.lower() in
                        {".wav", ".flac", ".ogg", ".mp3"})
            if ex:
                gr.Examples([[str(p)] for p in ex], inputs=audio_in, label="Example clips")
        btn.click(run, inputs=[audio_in, *sliders], outputs=[out_rt, out_base, out_steer])

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="Phase 2a control checkpoint.")
    ap.add_argument("--names", nargs="+", default=["loudness", "centroid", "harmonic_ratio"],
                    help="Descriptors the checkpoint was trained on (order matters).")
    ap.add_argument("--offset", type=int, default=None,
                    help="Prediction offset to render (default: model's smallest).")
    ap.add_argument("--max-seconds", type=float, default=8.0,
                    help="Trim input to this many seconds (0 = no trim).")
    ap.add_argument("--examples", type=Path, default=None, help="Dir of example clips for the UI.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Expose a public Gradio link.")
    args = ap.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        raise SystemExit("Gradio is required for the web demo. Install it with:\n"
                         '    pip install -e ".[demo]"')

    demo = build_demo(args)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
