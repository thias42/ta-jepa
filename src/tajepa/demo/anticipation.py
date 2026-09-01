"""Anticipation demo — the V-JEPA-style, decoder-free showcase of the causal JEPA.

From past context only, the causal model predicts the near future of audio *in latent
space*; this builds a Gradio UI that plots its per-frame prediction error against
reference predictors under a spectrogram, marks the surprising frames, and reports
forecasting skill. A playhead sweeps both panels in time with the audio. Nothing here
touches the codec decoder — the claim is made in representation space, the same
discipline as V-JEPA.

Two references are shown. **Persistence** ("next = now") is the intuitive one, but it is a
very weak bar on temporally smooth latents. The honest bar is a **linear AR(p)** fit in
closed form on the same latents (``ar_corpus``): the causal predictor has to beat *that*
for "it learned dynamics" to mean anything. Both skills are reported; the AR one counts.

``build_anticipation_demo`` takes already-constructed ``jepa`` / ``target`` (EMA encoder) /
``codec`` so it is agnostic to *how* the checkpoint was loaded — the local CLI
(`scripts/demo_anticipation.py`) loads a Lightning checkpoint; the HF Space loads weights
directly. Gradio/matplotlib/scipy are imported lazily (only the ``demo`` extra needs them).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch

# Axes occupy this horizontal fraction of the figure (set via subplots_adjust); the JS
# playhead maps audio time -> pixels with the same numbers so it tracks the data area, not
# the image edges.
_AX_LEFT, _AX_RIGHT = 0.07, 0.98
_PH_LEFT_PCT, _PH_SPAN_PCT = _AX_LEFT * 100, (_AX_RIGHT - _AX_LEFT) * 100

# Inject once into the page <head> (pass to ``demo.launch(head=HEAD_JS)``): a rAF loop that
# moves each player's playhead to its audio's currentTime. Runs continuously so it picks up
# players rendered on each analyze.
HEAD_JS = """
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
    """Self-contained player: figure image + overlaid playhead + audio element. Only static
    elements live here (the head rAF loop animates the playhead), so it survives any HTML
    sanitization."""
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


def _mel_db(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Log-mel spectrogram aligned to the codec frame rate (hop = sr / frame_rate)."""
    import librosa

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=80)
    return librosa.power_to_db(S, ref=np.max)


def build_anticipation_demo(jepa, target, codec, *, max_seconds: float = 12.0,
                            examples: str | Path | None = None,
                            ar_state: str | Path | None = None,
                            ar_corpus: str | Path | None = None, ar_order: int = 4,
                            context_frames: int | None = None):
    """Build (but don't launch) the anticipation Gradio ``Blocks``.

    ``jepa`` / ``target`` should already be in eval mode; all three of ``jepa`` / ``target``
    / ``codec`` should be on the same device. Launch with ``demo.launch(head=HEAD_JS, ...)``.

The linear-AR reference comes from one of two places, in order of preference:

    * ``ar_state`` — a ``LinearAR`` fitted offline on the model's *training* distribution
      and saved with ``LinearAR.save``. This is the right way: the reference matches the
      one the reported numbers use, costs nothing at startup, and does not depend on which
      clips happen to be bundled.
    * ``ar_corpus`` — a directory of audio to fit from at startup (defaults to
      ``examples``). A fallback. Prefer a corpus disjoint from the clips being analyzed:
      fitting the AR on the very clip under analysis flatters it, badly on periodic signals.

    With neither, the AR curve is omitted and only the weak persistence skill is reported,
    clearly labelled as such.

    ``context_frames`` is the sequence length the checkpoint was trained on; it defaults to
    the value recorded on the model, else 256. Audio longer than that is run through
    overlapping windows (``windowed_predict``) rather than in one pass. Without this the
    demo silently plots *positional extrapolation* past the training length: absolute
    sinusoidal encodings are defined at every index but were never learned there, and the
    prediction degrades to worse than persistence — on a 256-frame model, at exactly 3.41 s.
    """
    import gradio as gr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    from ..data.io import load_resampled
    from ..models.jepa import windowed_predict

    device = next(jepa.parameters()).device
    ctx = context_frames or getattr(jepa, "trained_context", None) or 256
    if context_frames is None and getattr(jepa, "trained_context", None) is None:
        print(f"[ta-jepa] context window: {ctx} frames (assumed — the checkpoint records "
              f"none). Audio beyond {ctx / codec.frame_rate:.2f}s is run in overlapping "
              f"windows; set context_frames if the model was trained differently.")
    else:
        print(f"[ta-jepa] context window: {ctx} frames "
              f"({ctx / codec.frame_rate:.2f}s); longer audio runs in overlapping windows")
    offsets = tuple(jepa.offsets)
    sr = int(codec.sample_rate)
    hop = max(1, round(sr / codec.frame_rate))       # samples per codec frame (≈320 @ 24k/75)
    _cache: dict = {}

    @torch.no_grad()
    def _fit_latent_ar():
        """Closed-form AR(p) on target latents — the reference the model must actually beat.

        Announces on stdout which reference is in use. Losing the AR degrades the demo
        silently — it just stops reporting the only number that discriminates and falls back
        to the flattering one — so the startup log has to say which of the three cases it is.
        """
        from ..models.linear_ar import LinearAR

        if ar_state:
            ar = LinearAR.load(ar_state).eval()
            print(f"[ta-jepa] AR reference: linear AR({ar.order}) loaded from {ar_state}")
            return ar
        root = Path(ar_corpus or examples) if (ar_corpus or examples) else None
        clips = sorted(p for p in root.glob("*") if p.suffix.lower() in
                       {".wav", ".flac", ".ogg", ".mp3"}) if root else []
        if not clips:
            print("[ta-jepa] AR reference: NONE — reporting skill vs persistence only, which "
                  "is a weak bar on smooth latents. Pass ar_state (preferred) or ar_corpus.")
            return None
        seqs = []
        for c in clips:
            wav = load_resampled(str(c), sr, mono=True)[:, : int(max_seconds * sr)]
            # truncate to the context: latents past it are extrapolation, and fitting the
            # reference on them would measure the model's failure rather than the audio
            seqs.append(target(codec.encode(wav.unsqueeze(0).to(device)).to(device)[:, :ctx]).cpu())
        dim = seqs[0].shape[-1]
        ar = LinearAR(dim, order=ar_order, offsets=offsets).fit(seqs)
        print(f"[ta-jepa] AR reference: linear AR({ar_order}) fitted on {len(clips)} clips "
              f"from {root} — a fallback; prefer ar_state fitted on the training distribution.")
        return ar

    _ar = _fit_latent_ar()
    _ar_p = _ar.order if _ar is not None else ar_order

    @torch.no_grad()
    def _ingest(path: str):
        key = (path, max_seconds)
        if key in _cache:
            return _cache[key]
        wav = load_resampled(path, sr, mono=True)            # [1, N]
        if max_seconds:
            wav = wav[:, : int(max_seconds * sr)]
        y = wav.squeeze(0).cpu().numpy()
        x = codec.encode(wav.unsqueeze(0).to(device)).to(device)   # [1, T, D] raw
        # Windowed, not one pass: past the trained context the positional encodings are
        # unlearned and the forecast collapses below persistence (see windowed_predict).
        preds, z_tgt = windowed_predict(jepa, target, x, context=ctx, stride=max(1, ctx // 2))
        mel = _mel_db(y, sr, hop)
        out = (y, preds, z_tgt, mel, x.shape[1])
        _cache[key] = out
        return out

    def _curves(preds, z_tgt, k: int):
        """Per-frame model / persistence / AR latent error, aligned to event time τ = k..T-1."""
        T = z_tgt.shape[1]
        model = (preds[k][:, : T - k] - z_tgt[:, k:]).abs().mean(-1).squeeze(0)   # [T-k]
        persist = (z_tgt[:, k:] - z_tgt[:, : T - k]).abs().mean(-1).squeeze(0)    # [T-k]
        ar = None
        if _ar is not None:
            p = _ar(z_tgt.cpu())[k].to(z_tgt)[:, : T - k]
            ar = (p - z_tgt[:, k:]).abs().mean(-1).squeeze(0).cpu().numpy()
        t = np.arange(k, T) / codec.frame_rate
        return t, model.cpu().numpy(), persist.cpu().numpy(), ar

    def analyze(path, k):
        if not path:
            return "", "Load a clip to analyze."
        k = int(k)
        y, preds, z_tgt, mel, T = _ingest(path)
        if T <= k + 1:
            return "", f"Clip too short for horizon k={k}."
        t, model, persist, ar = _curves(preds, z_tgt, k)
        dur = len(y) / sr
        skill = 1.0 - float(model.mean()) / max(float(persist.mean()), 1e-8)
        ar_skill = (1.0 - float(model.mean()) / max(float(ar.mean()), 1e-8)
                    if ar is not None else None)

        # top surprise peaks (prominent local maxima of the model error)
        prom = (model.max() - model.min()) * 0.15 + 1e-9
        peaks, _ = find_peaks(model, distance=max(1, int(0.15 * codec.frame_rate)), prominence=prom)
        peaks = peaks[np.argsort(model[peaks])[::-1][:6]] if len(peaks) else peaks

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1.4]})
        ax1.imshow(mel, origin="lower", aspect="auto", extent=[0, dur, 0, mel.shape[0]],
                   cmap="magma")
        ax1.set_ylabel("mel bin")
        title = (f"Anticipation — horizon k={k} ({k / codec.frame_rate * 1000:.0f} ms ahead)"
                 f"   ·   skill vs persistence: {skill:+.1%}")
        if ar_skill is not None:
            title += f"   ·   vs linear AR({_ar_p}): {ar_skill:+.1%}"
        ax1.set_title(title)
        for p in peaks:
            ax1.axvline(t[p], color="cyan", lw=1.0, alpha=0.7)

        ax2.plot(t, persist, color="gray", lw=1.2, label="persistence (next = now)")
        if ar is not None:
            ax2.plot(t, ar, color="tab:blue", lw=1.2, ls="--",
                     label=f"linear AR({_ar_p}) — the real bar")
        ax2.plot(t, model, color="crimson", lw=1.4, label="model prediction error")
        ax2.fill_between(t, model, persist, where=(persist >= model), color="crimson",
                         alpha=0.12, interpolate=True)
        ax2.scatter(t[peaks], model[peaks], color="cyan", zorder=5, s=24, label="surprise peaks")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("latent error (L1)")
        ax2.set_xlim(0, dur); ax2.legend(loc="upper right", fontsize=8)
        # explicit margins (not tight_layout) so the data area maps to the fractions the JS
        # playhead assumes
        fig.subplots_adjust(left=_AX_LEFT, right=_AX_RIGHT, top=0.92, bottom=0.11, hspace=0.07)

        msg = (f"**Skill vs persistence:** {skill:+.1%} — beats *assume-nothing-changes*, which "
               f"is an easy bar on smooth latents.")
        if ar_skill is not None:
            msg += (f"  **Skill vs linear AR({_ar_p}):** {ar_skill:+.1%} — the bar that counts: "
                    f"a closed-form linear fit on the last {_ar_p} latents, no training. "
                    f"Negative means the trained predictor does not beat it.")
        msg += ("  **Surprise peaks** (cyan) mark the frames the model found least predictable "
                "from the past — press play and watch the playhead.")
        html = _player_html(fig, y, sr)
        plt.close(fig)
        return html, msg

    with gr.Blocks(title="ta-jepa anticipation") as demo:   # head JS injected at launch()
        gr.Markdown(
            "# ta-jepa — anticipation\n"
            "A causal audio **world model**: from past context only, it predicts the near "
            "future of the sound *in latent space* (never decoding audio — the same discipline "
            "as V-JEPA). The plot shows its per-frame prediction error against two "
            "references: **persistence** (gray, 'assume nothing changes') and a **linear AR** "
            "fit (blue dashed). Persistence is an easy bar on smooth latents — the AR curve "
            "is the one that counts. **Spikes** mark surprising events. Press "
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
        if examples:
            ex = sorted(p for p in Path(examples).glob("*") if p.suffix.lower() in
                        {".wav", ".flac", ".ogg", ".mp3"})
            if ex:
                gr.Examples([[str(p)] for p in ex], inputs=audio_in, label="Example clips")
        btn.click(analyze, inputs=[audio_in, horizon], outputs=[out_player, out_msg])

    return demo
