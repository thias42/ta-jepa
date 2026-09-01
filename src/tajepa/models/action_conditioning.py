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


def action_windows(action: torch.Tensor, offsets) -> dict[int, torch.Tensor]:
    """``action [B,T,A]`` -> ``{offset: [B,T,A]}``, summed over the half-open window
    ``[t, t+offset)``.

    That is the standard convention — ``a_t`` is the action that takes the world from state
    ``t`` to state ``t+1`` — so predicting ``z_{t+o}`` uses ``a_t … a_{t+o-1}``. Windows
    running past the end are truncated, and those positions are dropped by the prediction
    loss anyway.
    """
    b, t, a = action.shape
    cum0 = torch.cat([action.new_zeros(b, 1, a), torch.cumsum(action, dim=1)], dim=1)
    idx = torch.arange(t, device=action.device)
    return {int(o): cum0[:, torch.clamp(idx + int(o), max=t)] - cum0[:, :t] for o in offsets}
