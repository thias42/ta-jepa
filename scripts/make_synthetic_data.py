"""Generate a tiny synthetic multi-domain audio set.

This exists so the whole Phase 0 pipeline (manifest -> codec embeddings -> APC
training) is runnable end-to-end *today*, before any real dataset (AudioSet /
FMA / ESC-50) is downloaded. The clips are crude stand-ins for the three domains
the model targets — tonal/"music", noisy/"environmental", formant-ish "speech" —
purely to exercise shapes and the training loop. They are NOT for evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _sine(t, f):
    return np.sin(2 * np.pi * f * t)


def make_clip(kind: str, seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    if kind == "music":          # stacked harmonics + slow vibrato (tonal/polyphonic-ish)
        f0 = rng.uniform(110, 330)
        vib = 1 + 0.01 * _sine(t, rng.uniform(3, 6))
        x = sum(_sine(t, f0 * k * vib) / k for k in (1, 2, 3, 4))
    elif kind == "environmental":  # filtered noise bursts (texture/transients)
        x = rng.standard_normal(t.shape)
        env = (np.sin(2 * np.pi * rng.uniform(1, 4) * t) > 0.5).astype(float)
        x = x * (0.2 + env)
    else:                         # "speech": formant-like AM of a buzz source
        f0 = rng.uniform(90, 180)
        buzz = np.sign(_sine(t, f0))
        formant = _sine(t, rng.uniform(600, 1200)) * 0.5 + 0.5
        x = buzz * formant
    x = x / (np.max(np.abs(x)) + 1e-8) * 0.9
    return x.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--per-domain", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for domain in ("music", "environmental", "speech"):
        d = args.out / domain
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_domain):
            x = make_clip(domain, args.seconds, args.sr, rng)
            sf.write(d / f"{domain}_{i:03d}.wav", x, args.sr)
    print(f"Wrote {3 * args.per_domain} clips to {args.out}/")


if __name__ == "__main__":
    main()
