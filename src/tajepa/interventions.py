"""Phase 2 — exogenous acoustic interventions (the *actions*).

Descriptors are not actions: they describe the very observation being predicted, which is
conditional generation, and is why the Phase 2b codebook could only collapse onto energy.
An action has to be something *applied* to the world (or to the sensing of it), with a
response the model must learn. Here the actions are motivated by robot audition:

- **gain** — the microphone is covered, or the distance to the source changes;
- **tilt** — the microphone is occluded, so the signal is muffled (lowpass);
- **reverb** — the robot moves into a different room (RT60 change).

These are *observation-model* interventions: covering a microphone changes how the scene is
sensed, not the scene. That is the audio analogue of camera pose in V-JEPA-2-AC, and for a
mobile robot ego-motion is the dominant agent-controlled cause of audio change — so it is the
right analogue, and the claim it supports is "action-conditioned audio prediction".

Two design rules, both learned expensively in Phase 2a:

1. **Interventions are timed events, not per-clip labels.** A uniformly reverberant clip
   teaches a static transform; reverb switching on mid-clip forces the model to predict a
   *transition*, which is what makes this action-conditioned dynamics rather than style
   transfer.
2. **Level is re-normalized after tilt and reverb, and gain is applied afterwards.** Lowpass
   removes energy and reverb adds a tail, so without this every axis would correlate with
   loudness and the loudness axis would be rediscovered for the fourth time. After
   normalization, *gain is the only axis that changes level*.

The action handed to the model is the **per-frame delta of the commanded parameter** — zero
away from the transition, non-zero across it. That mirrors a robot's per-timestep command
(a joint velocity, not a joint position), and it keeps the semantics "an action occurred"
rather than "the observation currently looks like this".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

#: Order of the action vector's axes. Values are physical deltas (dB, octaves, seconds);
#: standardize at training time as descriptors are.
AXES = ("gain_db", "tilt_oct", "rt60_s")


@dataclass(frozen=True)
class InterventionSpec:
    """One applied intervention: what changed, when, and how fast.

    ``event_s`` is where the transition starts and ``ramp_s`` how long it takes. The ramp is
    short but non-zero — a hard switch would click, and a real occlusion or doorway crossing
    is fast rather than instantaneous.
    """

    event_s: float
    ramp_s: float = 0.08
    gain_db: float = 0.0          # post-event level change (the only level-changing axis)
    cutoff_hz_before: float = 0.0  # 0 = no lowpass
    cutoff_hz_after: float = 0.0
    rt60_before: float = 0.0      # 0 = anechoic
    rt60_after: float = 0.0

    @property
    def tilt_oct(self) -> float:
        """Cutoff change in octaves — 0 when either side is unfiltered (not comparable)."""
        if self.cutoff_hz_before <= 0 or self.cutoff_hz_after <= 0:
            return 0.0
        return float(np.log2(self.cutoff_hz_after / self.cutoff_hz_before))

    @property
    def deltas(self) -> dict[str, float]:
        return {"gain_db": self.gain_db, "tilt_oct": self.tilt_oct,
                "rt60_s": self.rt60_after - self.rt60_before}


def synthetic_rir(rt60: float, sr: int, rng: np.random.Generator,
                  damping: float = 0.35) -> np.ndarray:
    """Exponentially-decaying noise burst with the requested RT60, unit energy.

    A parametric stand-in for a measured impulse response: dependency-free, and RT60 is a
    continuous knob, which is what makes it usable as an action parameter. High frequencies
    decay faster (``damping``), as in a real room. Real or `pyroomacoustics`-generated RIRs
    should be mixed in before any claim about generalization — a model trained only on this
    can learn to detect *this* decay shape.
    """
    if rt60 <= 0:
        return np.array([1.0], dtype=np.float32)
    n = max(8, int(rt60 * sr))
    t = np.arange(n) / sr
    ir = rng.normal(size=n).astype(np.float32) * np.exp(-6.9078 * t / rt60).astype(np.float32)
    if damping > 0:                       # highs die sooner than lows
        b, a = signal.butter(1, min(0.99, damping), btype="low")
        ir = signal.lfilter(b, a, ir).astype(np.float32)
    ir[0] += 1.0                          # keep a direct path
    e = float(np.sqrt(np.sum(ir ** 2)))
    return (ir / e).astype(np.float32) if e > 0 else ir


def _lowpass(x: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return x
    b, a = signal.butter(order, cutoff_hz / (sr / 2), btype="low")
    return signal.filtfilt(b, a, x).astype(np.float32)


def _reverb(x: np.ndarray, rt60: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    if rt60 <= 0:
        return x
    return signal.fftconvolve(x, synthetic_rir(rt60, sr, rng))[: len(x)].astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _crossfade(before: np.ndarray, after: np.ndarray, start: int, ramp: int) -> np.ndarray:
    """Equal-power crossfade from ``before`` to ``after`` over ``ramp`` samples at ``start``."""
    out = before.copy()
    n = len(out)
    lo, hi = max(0, min(start, n)), max(0, min(start + ramp, n))
    if hi > lo:
        w = np.linspace(0.0, 1.0, hi - lo, dtype=np.float32)
        out[lo:hi] = np.cos(w * np.pi / 2) * before[lo:hi] + np.sin(w * np.pi / 2) * after[lo:hi]
    out[hi:] = after[hi:]
    return out


def apply_intervention(
    wav: np.ndarray, sr: int, spec: InterventionSpec, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render the intervened signal: two full renders crossfaded at the event.

    Rendering both sides over the whole clip and crossfading is deliberate — it is how moving
    between rooms actually sounds (the tail of the old room persists into the new one), and it
    avoids the transient a mid-stream filter-state switch would introduce.
    """
    rng = rng or np.random.default_rng(0)
    x = wav.astype(np.float32)
    ref = _rms(x)

    def render(cutoff: float, rt60: float, seed: int) -> np.ndarray:
        y = _reverb(_lowpass(x, cutoff, sr), rt60, sr, np.random.default_rng(seed))
        # rule 2: strip the level side-effect of tilt/reverb so gain owns the level axis
        r = _rms(y)
        return y * (ref / r) if r > 0 else y

    before = render(spec.cutoff_hz_before, spec.rt60_before, 1)
    after = render(spec.cutoff_hz_after, spec.rt60_after, 2)
    after = after * float(10.0 ** (spec.gain_db / 20.0))     # the one deliberate level change
    return _crossfade(before, after, int(spec.event_s * sr), max(1, int(spec.ramp_s * sr)))


def action_frames(spec: InterventionSpec, n_frames: int, frame_rate: float) -> np.ndarray:
    """Per-frame action ``[n_frames, len(AXES)]``: the commanded parameter *delta* per frame.

    Zero away from the transition; across the ramp each axis carries its total change spread
    over the ramp's frames, so the row sum equals the commanded delta. A step intervention is
    therefore a brief spike, and continuous automation (a moving robot) would be a sustained
    non-zero signal — the same encoding covers both.
    """
    a = np.zeros((n_frames, len(AXES)), dtype=np.float32)
    start = int(spec.event_s * frame_rate)
    ramp = max(1, int(round(spec.ramp_s * frame_rate)))
    lo, hi = max(0, min(start, n_frames)), max(0, min(start + ramp, n_frames))
    if hi > lo:
        d = spec.deltas
        for j, name in enumerate(AXES):
            a[lo:hi, j] = d[name] / (hi - lo)
    return a


def transient_score(wav: np.ndarray, sr: int, hop: int = 320) -> float:
    """Cheap measure of how impulsive a signal is: mean positive spectral flux of the frame
    energy, normalised by level. Used to decide whether a reverb action is *observable*."""
    n = max(1, len(wav) // hop)
    e = np.abs(wav[: n * hop].reshape(n, hop)).mean(1) + 1e-8
    d = np.diff(np.log(e))
    return float(np.clip(d, 0, None).mean())


def sample_spec(
    rng: np.random.Generator, duration_s: float, *,
    p_axis: float = 0.6, edge_s: float = 0.5,
    transient: float | None = None, reverb_min_transient: float = 0.05,
) -> InterventionSpec:
    """Draw a random intervention. Each axis fires independently with prob ``p_axis``.

    Independent draws (rather than one axis at a time) mean the training set contains
    combinations, so the model cannot succeed by learning three disjoint detectors — and it
    lets the eval ask whether the axes compose.

    **Reverb is gated on content observability.** A room is heard through its response to
    transients; on stationary texture (rain, wind, fire) a reverb change is close to
    physically invisible. Measured on the paired cache, the applied RT60 is recoverable at
    R²≈+0.22 on transient-rich clips and −2.0 on stationary ones. Commanding an action with
    no observable consequence is exactly what trains a predictor to ignore that axis — it is
    how the Phase 2a onset dial died — so when ``transient`` is supplied and falls below
    ``reverb_min_transient``, the reverb axis simply does not fire.
    """
    lo, hi = edge_s, max(edge_s + 0.1, duration_s - edge_s)
    spec: dict = {"event_s": float(rng.uniform(lo, hi)),
                  "ramp_s": float(rng.uniform(0.04, 0.20))}
    if rng.random() < p_axis:                                  # microphone covered / distance
        spec["gain_db"] = float(rng.uniform(-18.0, 6.0))
    if rng.random() < p_axis:                                  # occluded -> muffled
        before = float(rng.uniform(6000, 11000))
        spec["cutoff_hz_before"] = before
        spec["cutoff_hz_after"] = float(before * 2 ** rng.uniform(-3.0, 0.5))
    reverb_ok = transient is None or transient >= reverb_min_transient
    if reverb_ok and rng.random() < p_axis:                    # walked into another room
        spec["rt60_before"] = float(rng.uniform(0.05, 0.4))
        spec["rt60_after"] = float(rng.uniform(0.05, 1.2))
    return InterventionSpec(**spec)
