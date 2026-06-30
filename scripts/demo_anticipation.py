"""Web demo — temporal *anticipation* of a Phase 1 causal JEPA (the V-JEPA analogue).

Thin CLI wrapper: loads a Lightning checkpoint and launches the shared anticipation demo
(`tajepa.demo.anticipation`). Showcases what the model is *validated* to do — predict the
near future of audio in latent space and beat a persistence baseline — without ever
touching the (lossy) decoder. See the package module for the methodology.

    python scripts/demo_anticipation.py --ckpt runs/jepa_fma_grounded.ckpt \
        --examples data/demo_clips

Requires the demo extra:  pip install -e ".[demo]"   (adds gradio + matplotlib)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.config import CodecConfig, resolve_device
from tajepa.codec.frontend import build_frontend
from tajepa.demo.anticipation import HEAD_JS, build_anticipation_demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="Phase 1 causal JEPA checkpoint.")
    ap.add_argument("--max-seconds", type=float, default=12.0, help="Trim input (0 = no trim).")
    ap.add_argument("--examples", type=Path, default=None, help="Dir of example clips for the UI.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true", help="Expose a public Gradio link.")
    args = ap.parse_args()

    try:
        import gradio  # noqa: F401
        import matplotlib  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"Missing demo dependency ({e.name}). Install with:\n"
                         '    pip install -e ".[demo]"')

    device = args.device or resolve_device("auto")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_jepa import JEPALightning

    # strict=False so checkpoints that predate the grounding head still load — the
    # anticipation demo lives entirely in latent space and never uses recon_head.
    lit = JEPALightning.load_from_checkpoint(str(args.ckpt), map_location="cpu", strict=False)
    jepa = lit.jepa.eval().to(device)
    target = lit.target.eval().to(device)
    codec = build_frontend(CodecConfig(device=device))

    demo = build_anticipation_demo(jepa, target, codec, max_seconds=args.max_seconds,
                                   examples=args.examples)
    allowed = [str(Path(args.examples).resolve())] if args.examples else None
    # Gradio 6 takes `head` (the playhead-animation script) on launch(), not on Blocks().
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=allowed, head=HEAD_JS)


if __name__ == "__main__":
    main()
