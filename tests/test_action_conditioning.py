import torch

from tajepa.eval.intervention import counterfactual_report
from tajepa.interventions import InterventionSpec, action_frames
from tajepa.models.action_conditioning import action_windows


def test_action_window_uses_the_standard_convention():
    """a_t is the action taking the world from t to t+1, so predicting z_{t+o} uses
    a_t .. a_{t+o-1} — the half-open window [t, t+o)."""
    a = torch.zeros(1, 12, 1)
    a[0, 5, 0] = 3.0
    w = action_windows(a, (1, 2, 4))
    assert w[1][0, 5, 0] == 3.0 and w[1][0, 4, 0] == 0.0 and w[1][0, 6, 0] == 0.0
    assert all(w[4][0, t, 0] == 3.0 for t in (2, 3, 4, 5))
    assert w[4][0, 1, 0] == 0.0 and w[4][0, 6, 0] == 0.0


def test_action_window_sums_a_ramp_to_the_total_commanded_change():
    spec = InterventionSpec(event_s=1.0, ramp_s=0.1, gain_db=-9.0)
    a = torch.from_numpy(action_frames(spec, 200, 75.0))[None]
    w = action_windows(a, (32,))[32]
    assert float(w[0, :, 0].min()) == torch.tensor(-9.0).item() or \
        abs(float(w[0, 60, 0]) - (-9.0)) < 1e-3       # a window spanning the whole ramp


def test_windows_are_zero_when_no_action_is_applied():
    a = torch.zeros(2, 20, 3)
    for o, w in action_windows(a, (1, 4)).items():
        assert w.shape == (2, 20, 3) and not w.any()


class _Fake(torch.nn.Module):
    """Predicts the target only when told the action; ignores it if ``deaf``."""

    def __init__(self, deaf: bool):
        super().__init__()
        self.offsets = (1,)
        self.cond_dim = 1
        self.deaf = deaf

    def predict_with_deltas(self, x, deltas, desc=None, pad_mask=None):
        d = deltas[1]
        base = torch.zeros(x.shape[0], x.shape[1], 4)
        if not self.deaf:
            base = base + d[..., :1]          # uses the commanded action
        return None, {1: base}


class _Target(torch.nn.Module):
    def forward(self, x):
        return x[..., :4]


class _DS:
    def __init__(self, action, target):
        self.a, self.t = action, target

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"intervened": self.t, "action": self.a}


def test_counterfactual_detects_action_use_and_its_absence():
    """The eval must separate a model that uses the action from one that ignores it —
    otherwise a dead dial would look identical to a working one, which is exactly how the
    Phase 2a onset control went unnoticed."""
    t_len = 24
    action = torch.zeros(t_len, 1)
    action[10, 0] = 2.0
    # The fake model can only express the action inside its own prediction window, so the
    # target is the impulse that window conveys — enough to test that the eval separates a
    # listening model from a deaf one, which is all this fixture is for.
    target = torch.zeros(t_len, 4)
    target[11] = 2.0

    ds = _DS(action, target)
    listening = counterfactual_report(_Fake(deaf=False), _Target(), ds, offsets=(1,),
                                      device="cpu", axis_names=("gain_db",))
    deaf = counterfactual_report(_Fake(deaf=True), _Target(), ds, offsets=(1,),
                                 device="cpu", axis_names=("gain_db",))
    assert listening["overall"][1]["action_gain"] > 0.5
    assert abs(deaf["overall"][1]["action_gain"]) < 1e-6
    # the two predictions must be scored against the same target, so err_without matches
    assert listening["overall"][1]["err_without"] == deaf["overall"][1]["err_without"]


def test_counterfactual_attributes_per_axis_only_on_single_axis_clips():
    t_len = 24
    action = torch.zeros(t_len, 2)
    action[10] = torch.tensor([1.0, 1.0])          # two axes fire together
    ds = _DS(action, torch.zeros(t_len, 4))
    rep = counterfactual_report(_Fake(deaf=False), _Target(), ds, offsets=(1,),
                                device="cpu", axis_names=("gain_db", "tilt_oct"))
    assert rep["overall"], "overall should still be reported"
    assert rep["per_axis"] == {}, "no axis may claim credit when two fired together"
