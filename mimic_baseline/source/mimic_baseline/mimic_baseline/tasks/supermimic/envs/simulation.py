"""Shared simulation defaults; simulator imports occur only at construction."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from mimic_baseline.assets import ASSET_DIR

PHYSICS_DT = 0.005
GROUND_PLANE_USD = str(
    Path(ASSET_DIR) / "terrains/isaac_sim_6_grid/default_environment.usd"
)


def make_sim(device: str, args: Any = None):
    """Create the RX-P2 PhysX simulation config at an explicit 200 Hz."""
    del args
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}.")
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxCfg

    return sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        physics=PhysxCfg(),
    )
