"""Shared network-action range for simulation, imagination and deployment."""

import torch


ACTION_LIMIT = 20.0


def clip_action(action: torch.Tensor) -> torch.Tensor:
    """Bound the command without changing values or gradients inside the range."""
    return action.clamp(-ACTION_LIMIT, ACTION_LIMIT)
