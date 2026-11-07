"""Shared tensor helpers for the original learning loop."""

from __future__ import annotations


import torch


import torch.nn as nn


def _valid_depth(depth: torch.Tensor) -> torch.Tensor:
    """Recover the stereo validity mask from a stored depth frame.

    Invalid pixels are written as a negative sentinel, so a stored frame is
    self-describing at any range. A distance threshold would not work: the
    sensor's ``depth_limit`` is 8 m while the actor observation clips to 2 m, so
    a real surface at 3 m would be indistinguishable from a miss. Storing the
    mask separately would double the depth traffic through replay.
    """
    return depth >= 0.0


def _as_modules(modules: nn.Module | tuple[nn.Module, ...]) -> tuple[nn.Module, ...]:
    return modules if isinstance(modules, tuple) else (modules,)
