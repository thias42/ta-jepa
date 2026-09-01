"""Turning a per-frame action stream into per-offset predictor conditioning.

The action written by ``interventions.action_frames`` is a per-frame *delta* of the commanded
parameter — a robot's per-timestep command, not a state description. To predict ``z_{t+o}``
from ``z_t`` the predictor needs to know what was commanded **in between**, so the
conditioning at offset ``o`` is the cumulative action over ``(t, t+o]``.

That is the only sensible reading, and it is what makes the setup action-conditioned rather
than observation-conditioned: away from an event the conditioning is zero and the model must
fall back on dynamics; across one it carries exactly the intervention the model has to
anticipate.
"""

from __future__ import annotations

import torch


def action_windows(action: torch.Tensor, offsets, lead_frames: int = 0) -> dict[int, torch.Tensor]:
    """``action [B,T,A]`` -> ``{offset: [B,T,A]}``, summed over ``[t−L, t+offset−L)``.

    With no lead this is the standard convention — ``a_t`` takes the world from state ``t`` to
    ``t+1``, so predicting ``z_{t+o}`` uses ``a_t … a_{t+o-1}``.

    ``lead_frames`` (``L``) matches the delay between commanding an action and its acoustic
    effect (``InterventionSpec.lead_s``). If the effect of ``a_f`` lands at ``f+L+1``, then the
    actions determining ``z_{t+o}`` are those at ``t−L … t+o−L−1``, so the window shifts back
    by ``L``. **Shifting the data without shifting this window is worse than no lead at all**:
    the action would pass out of the conditioning window before the frame it explains, and the
    model would be handed zeros exactly where the information lives.

    Windows running past either end are truncated; those positions are dropped by the
    prediction loss anyway.
    """
    b, t, a = action.shape
    cum0 = torch.cat([action.new_zeros(b, 1, a), torch.cumsum(action, dim=1)], dim=1)
    idx = torch.arange(t, device=action.device)
    lead = int(lead_frames)
    lo = torch.clamp(idx - lead, min=0, max=t)
    return {int(o): cum0[:, torch.clamp(idx + int(o) - lead, min=0, max=t)] - cum0[:, lo]
            for o in offsets}
