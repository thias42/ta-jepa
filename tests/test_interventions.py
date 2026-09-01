import numpy as np
import pytest

from tajepa.interventions import (
    AXES, InterventionSpec, action_frames, apply_intervention, sample_spec,
    synthetic_rir, transient_score,
)

SR = 24000


def _tone(sec=4.0, f=440.0):
    t = np.arange(int(SR * sec)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_action_frames_sum_to_the_commanded_delta():
    spec = InterventionSpec(event_s=1.0, ramp_s=0.1, gain_db=-9.0,
                            cutoff_hz_before=8000, cutoff_hz_after=2000,
                            rt60_before=0.1, rt60_after=0.6)
    a = action_frames(spec, 300, 75.0)
    assert a.shape == (300, len(AXES))
    for j, name in enumerate(AXES):
        assert a[:, j].sum() == pytest.approx(spec.deltas[name], abs=1e-4)
    # zero away from the event: the action is "something happened", not a state description
    assert not a[:70].any() and not a[90:].any()


def test_gain_is_the_only_axis_that_changes_level():
    """Rule 2. Lowpass removes energy and reverb adds a tail; without post-normalisation
    every axis would correlate with loudness, which is how Phase 2a kept rediscovering it."""
    x = _tone()
    ev, rng = 2.0, np.random.default_rng(0)

    def level_change(spec):
        y = apply_intervention(x, SR, spec, rng)
        pre = np.sqrt((y[: int(1.8 * SR)] ** 2).mean())
        post = np.sqrt((y[int(2.4 * SR):] ** 2).mean())
        return 20 * np.log10(post / max(pre, 1e-9))

    assert abs(level_change(InterventionSpec(ev, cutoff_hz_before=9000,
                                             cutoff_hz_after=1500))) < 1.5
    assert abs(level_change(InterventionSpec(ev, rt60_before=0.05, rt60_after=0.8))) < 1.5
    assert level_change(InterventionSpec(ev, gain_db=-12.0)) == pytest.approx(-12.0, abs=1.5)


def test_intervention_is_a_timed_event_not_a_clip_label():
    """Before the event the signal must be untouched — otherwise the model learns a static
    transform rather than a transition, which is the whole distinction being drawn."""
    x = _tone()
    spec = InterventionSpec(event_s=2.0, ramp_s=0.05, gain_db=-20.0,
                            cutoff_hz_before=9000, cutoff_hz_after=1000)
    y = apply_intervention(x, SR, spec, np.random.default_rng(0))
    n = int(1.9 * SR)
    before = apply_intervention(x, SR, InterventionSpec(99.0, cutoff_hz_before=9000,
                                                        cutoff_hz_after=9000),
                                np.random.default_rng(0))
    assert np.abs(y[:n] - before[:n]).max() < 1e-4
    assert np.sqrt((y[int(2.5 * SR):] ** 2).mean()) < 0.2 * np.sqrt((y[:n] ** 2).mean())


def test_render_is_finite_and_length_preserving():
    x = _tone()
    for spec in (InterventionSpec(1.0, gain_db=6.0),
                 InterventionSpec(1.0, rt60_before=0.0, rt60_after=1.2),
                 InterventionSpec(1.0, cutoff_hz_before=10000, cutoff_hz_after=800)):
        y = apply_intervention(x, SR, spec, np.random.default_rng(0))
        assert y.shape == x.shape and np.isfinite(y).all()


def test_synthetic_rir_decays_and_has_unit_energy():
    rir = synthetic_rir(0.5, SR, np.random.default_rng(0))
    assert np.sum(rir ** 2) == pytest.approx(1.0, rel=1e-3)
    head = np.abs(rir[: len(rir) // 4]).mean()
    tail = np.abs(rir[-len(rir) // 4:]).mean()
    assert tail < head * 0.2                       # -60 dB over the window, so tail is small
    assert synthetic_rir(0.0, SR, np.random.default_rng(0)).size == 1   # anechoic


def test_reverb_only_fires_where_it_would_be_observable():
    """A room is heard through its response to transients. Commanding reverb on stationary
    texture is an action with no observable consequence — how the onset dial died."""
    rng = np.random.default_rng(1)
    steady = (0.1 * rng.normal(size=SR * 4)).astype(np.float32)
    t = np.arange(SR * 4) / SR
    impulsive = (np.sin(2 * np.pi * 300 * t) * np.exp(-((t % 0.5) * 12))).astype(np.float32)
    assert transient_score(steady, SR) < transient_score(impulsive, SR)

    def fired(w):
        ts = transient_score(w, SR)
        return sum(sample_spec(np.random.default_rng(i), 4.0, transient=ts).rt60_after > 0
                   for i in range(60))

    assert fired(steady) == 0
    assert fired(impulsive) > 10
    # gate off -> reverb fires regardless
    assert sum(sample_spec(np.random.default_rng(i), 4.0, transient=0.0,
                           reverb_min_transient=0.0).rt60_after > 0 for i in range(60)) > 10


def test_sampled_axes_are_independent():
    rng = np.random.default_rng(0)
    specs = [sample_spec(rng, 5.0, transient=1.0) for _ in range(400)]
    fires = np.array([[s.gain_db != 0, s.tilt_oct != 0, s.rt60_after > 0] for s in specs])
    assert (fires.mean(0) > 0.4).all() and (fires.mean(0) < 0.8).all()
    assert (fires.all(1)).mean() > 0.1      # combinations exist, not one axis at a time


def test_intervention_pair_dataset_round_trip(tmp_path):
    """The cache written by make_interventions.py must load frame-aligned, and cropping must
    land on the transition — a window drawn wholly before or after it has an all-zero action
    and teaches nothing about the response."""
    import torch

    from tajepa.data.embedding_dataset import InterventionPairDataset

    rng = np.random.default_rng(0)
    for i in range(4):
        spec = InterventionSpec(event_s=2.0, ramp_s=0.1, gain_db=-8.0)
        np.savez_compressed(
            tmp_path / f"c{i}.npz",
            clean=rng.normal(size=(400, 16)).astype(np.float32),
            intervened=rng.normal(size=(400, 16)).astype(np.float32),
            action=action_frames(spec, 400, 75.0),
        )
    ds = InterventionPairDataset(tmp_path, window_frames=128)
    assert len(ds) == 4
    item = ds[0]
    assert item["features"].shape == (128, 16)
    assert item["intervened"].shape == (128, 16)
    assert item["action"].shape == (128, len(AXES))
    hits = sum(int(ds[i % 4]["action"].abs().sum() > 0) for i in range(40))
    assert hits > 30, "event-centred cropping should usually include the transition"

    flat = InterventionPairDataset(tmp_path, window_frames=128, centre_on_event=False,
                                   random_crop=False)
    assert flat[0]["action"].abs().sum() == 0    # window 0..128, event at frame 150
