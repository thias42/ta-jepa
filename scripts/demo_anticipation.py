"""Web demo — temporal *anticipation* of a Phase 1 causal JEPA (the V-JEPA analogue).

This showcases what the model is *validated* to do: predict the near future of audio in
latent space and beat a persistence baseline. It never touches the (lossy) decoder — like
V-JEPA, the claim is made entirely in representation space.

For an input clip we run the causal encoder + predictor and the EMA target encoder, then
plot the model's **per-frame prediction error** over time against the **persistence**
baseline ("the future equals the present"), aligned under a spectrogram:

  * low error  → predictable stretches (sustained notes, steady texture),
  * error spikes → *surprising* events (onsets, an instrument/voice entering, scene cuts),
  * model curve sitting below persistence → the forecasting skill the project gates on.

Everything is in the model's own latent space (online prediction vs stop-grad EMA target),
so a temporally-smooth latent gets no free pass: persistence is smooth too, and we report
skill = 1 − mean(model error) / mean(persistence error).

    python scripts/demo_anticipation.py --ckpt runs/jepa_fma_grounded.ckpt \
        --examples data/demo_clips

Requires the demo extra:  pip install -e ".[demo]"   (adds gradio + matplotlib)
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import torch

# Axes occupy this horizontal fraction of the rendered figure (set via subplots_adjust);
# the JS playhead maps audio time -> pixels using the same numbers so it lines up with the
# data area, not the image edges.
_AX_LEFT, _AX_RIGHT = 0.07, 0.98
_PH_LEFT_PCT, _PH_SPAN_PCT = _AX_LEFT * 100, (_AX_RIGHT - _AX_LEFT) * 100

# Injected once into the page <head>: a rAF loop that moves the playhead of every player to
# its audio's currentTime. Runs continuously so it picks up players added on each analyze.
_HEAD_JS = """
<script>
(function () {
  function tick() {
    document.querySelectorAll('.ta-player').forEach(function (p) {
      var a = p.querySelector('audio'), ph = p.querySelector('.ta-ph');
      if (a && ph && a.duration) {
        var L = parseFloat(p.dataset.left), W = parseFloat(p.dataset.span);
        ph.style.left = (L + (a.currentTime / a.duration) * W) + '%';
      }
    });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
</script>
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    return base64.b64encode(buf.getvalue()).decode()


def _wav_to_b64(y: np.ndarray, sr: int) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _player_html(fig, y: np.ndarray, sr: int) -> str:
    """A self-contained player: figure image + overlaid playhead + audio element. The
    head-injected rAF loop animates the playhead; only static elements live here, so it
    survives HTML sanitization."""
    img = _fig_to_b64(fig)
    wav = _wav_to_b64(y, sr)
    return f"""
<div class="ta-player" data-left="{_PH_LEFT_PCT:.3f}" data-span="{_PH_SPAN_PCT:.3f}"
     style="position:relative;width:100%;max-width:900px;">
  <img src="data:image/png;base64,{img}" style="width:100%;display:block;"/>
  <div class="ta-ph" style="position:absolute;top:0;bottom:0;left:{_PH_LEFT_PCT:.3f}%;
       width:2px;background:#19e6ff;box-shadow:0 0 4px #19e6ff;pointer-events:none;"></div>
  <audio controls src="data:audio/wav;base64,{wav}" style="width:100%;margin-top:6px;"></audio>
</div>
"""

from tajepa.config import CodecConfig, resolve_device
from tajepa.codec.frontend import build_frontend
from tajepa.data.io import load_resampled


def _mel_db(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Log-mel spectrogram aligned to the codec frame rate (hop = sr / frame_rate)."""
    import librosa

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=80)
    return librosa.power_to_db(S, ref=np.max)


def build_demo(args):
    import gradio as gr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    device = args.device or resolve_device("auto")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_jepa import JEPALightning

    # strict=False so checkpoints that predate the grounding head still load — the
    # anticipation demo lives entirely in latent space and never uses recon_head.
    lit = JEPALightning.load_from_checkpoint(str(args.ckpt), map_location="cpu", strict=False)
    jepa = lit.jepa.eval().to(device)
    target = lit.target.eval().to(device)           # EMA target encoder (defines the targets)
    offsets = tuple(jepa.offsets)

    codec = build_frontend(CodecConfig(device=device))
    sr = int(codec.sample_rate)
    hop = max(1, round(sr / codec.frame_rate))       # samples per codec frame (≈320 @ 24k/75)
    _cache: dict = {}

    @torch.no_grad()
    def _ingest(path: str):
        key = (path, args.max_seconds)
        if key in _cache:
            return _cache[key]
        wav = load_resampled(path, sr, mono=True)            # [1, N]
        if args.max_seconds:
            wav = wav[:, : int(args.max_seconds * sr)]
        y = wav.squeeze(0).cpu().numpy()
        x = codec.encode(wav.unsqueeze(0).to(device))        # [1, T, D] raw
        _, preds = jepa(x)                                   # offset -> [1, T, dim]
        z_tgt = target(x)                                    # [1, T, dim] EMA targets
        mel = _mel_db(y, sr, hop)
        out = (y, preds, z_tgt, mel, x.shape[1])
        _cache[key] = out
        return out

    def _curves(preds, z_tgt, k: int):
        """Per-frame model vs persistence latent error, aligned to event time τ = k..T-1."""
        T = z_tgt.shape[1]
        model = (preds[k][:, : T - k] - z_tgt[:, k:]).abs().mean(-1).squeeze(0)   # [T-k]
        persist = (z_tgt[:, k:] - z_tgt[:, : T - k]).abs().mean(-1).squeeze(0)    # [T-k]
        t = np.arange(k, T) / codec.frame_rate
        return t, model.cpu().numpy(), persist.cpu().numpy()

    def analyze(path, k):
        if not path:
            return "", "Load a clip to analyze."
        k = int(k)
        y, preds, z_tgt, mel, T = _ingest(path)
        if T <= k + 1:
            return "", f"Clip too short for horizon k={k}."
        t, model, persist = _curves(preds, z_tgt, k)
        dur = len(y) / sr
        skill = 1.0 - float(model.mean()) / max(float(persist.mean()), 1e-8)

        # top surprise peaks (prominent local maxima of the model error)
        prom = (model.max() - model.min()) * 0.15 + 1e-9
        peaks, _ = find_peaks(model, distance=max(1, int(0.15 * codec.frame_rate)), prominence=prom)
        peaks = peaks[np.argsort(model[peaks])[::-1][:6]] if len(peaks) else peaks

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1.4]})
        ax1.imshow(mel, origin="lower", aspect="auto", extent=[0, dur, 0, mel.shape[0]],
                   cmap="magma")
        ax1.set_ylabel("mel bin")
        ax1.set_title(f"Anticipation — horizon k={k} ({k / codec.frame_rate * 1000:.0f} ms ahead)"
                      f"   ·   forecasting skill vs persistence: {skill:+.1%}")
        for p in peaks:
            ax1.axvline(t[p], color="cyan", lw=1.0, alpha=0.7)

        ax2.plot(t, persist, color="gray", lw=1.2, label="persistence (next = now)")
        ax2.plot(t, model, color="crimson", lw=1.4, label="model prediction error")
        ax2.fill_between(t, model, persist, where=(persist >= model), color="crimson",
                         alpha=0.12, interpolate=True)
        ax2.scatter(t[peaks], model[peaks], color="cyan", zorder=5, s=24, label="surprise peaks")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("latent error (L1)")
        ax2.set_xlim(0, dur); ax2.legend(loc="upper right", fontsize=8)
        # explicit margins (not tight_layout) so the data area maps to the fractions the JS
        # playhead assumes
        fig.subplots_adjust(left=_AX_LEFT, right=_AX_RIGHT, top=0.92, bottom=0.11, hspace=0.07)

        msg = (f"**Forecasting skill:** {skill:+.1%}  (1 − model/persistence error; >0 means it "
               f"beats *assume-nothing-changes*).  **Surprise peaks** (cyan) mark the frames the "
               f"model found least predictable from the past — press play and watch the playhead.")
        html = _player_html(fig, y, sr)
        plt.close(fig)
        return html, msg

    with gr.Blocks(title="ta-jepa anticipation") as demo:   # head JS injected at launch()
        gr.Markdown(
            "# ta-jepa — anticipation\n"
            "A causal audio **world model**: from past context only, it predicts the near "
            "future of the sound *in latent space* (never decoding audio — the same discipline "
            "as V-JEPA). The plot shows its per-frame prediction error vs a **persistence** "
            "baseline. Where the red curve dips below gray, the model is anticipating change "
            "that 'assume nothing changes' can't; **spikes** mark surprising events. Press "
            "play and the cyan **playhead** sweeps both panels in time with the audio."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(type="filepath", label="Input clip",
                                    sources=["upload", "microphone"])
                horizon = gr.Dropdown([str(o) for o in offsets], value=str(min(offsets)),
                                      label="Prediction horizon k (frames ahead, 75 fps)")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=2):
                out_player = gr.HTML(label="Spectrogram + anticipation error")
                out_msg = gr.Markdown()
        if args.examples:
            ex = sorted(p for p in Path(args.examples).glob("*") if p.suffix.lower() in
                        {".wav", ".flac", ".ogg", ".mp3"})
            if ex:
                gr.Examples([[str(p)] for p in ex], inputs=audio_in, label="Example clips")
        btn.click(analyze, inputs=[audio_in, horizon], outputs=[out_player, out_msg])

    return demo


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

    demo = build_demo(args)
    allowed = [str(Path(args.examples).resolve())] if args.examples else None
    # Gradio 6 takes `head` (the playhead-animation script) on launch(), not on Blocks().
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=allowed, head=_HEAD_JS)


if __name__ == "__main__":
    main()
