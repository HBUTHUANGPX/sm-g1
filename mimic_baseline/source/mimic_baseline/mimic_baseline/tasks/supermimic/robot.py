"""Paper-compatible RX-P2 articulation policy for standalone SuperMimic."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .kinematics import RXP2_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg


_ACTION_JOINT_NAMES = RXP2_JOINT_NAMES[:27]
_ACTUATOR_TYPES = frozenset(("implicit", "delayed_torque_speed"))


def make_robot_cfg(
    actuator_type: str = "delayed_torque_speed",
    self_collisions: bool = True,
    min_delay: int = 0,
    max_delay: int = 4,
    neck_yaw: float = 0.0,
    neck_pitch: float = 0.0,
    usd_path: str | None = None,
) -> tuple["ArticulationCfg", list[float]]:
    """Build the headed 29-joint RX-P2 with a 27-action body interface.

    The default is the RX-P2 hardware contract: explicit PD with hard
    torque-speed limiting and 0-4 physics substeps of command delay, i.e.
    0-20 ms at the 5 ms step. Action targets are

    ``q_default + 0.25 * effort_limit / stiffness * action``.

    ``implicit`` remains available for evaluating checkpoints trained with the
    paper-source controller structure (implicit PD, no delay).

    The paper leaves self-collision handling unstated. Enabling physical
    self-collisions is an RX-P2 adaptation inferred from the paper's discussion
    of reward hacking, not a claimed paper setting.

    ``neck_yaw`` and ``neck_pitch`` set the two neck joints' independent PD
    target in radians. They are held outside the 27-action policy and so are
    not part of ``action_scale``. The SuperMimic perception task uses
    ``neck_pitch = +0.5061 rad`` (29° downward tilt, +Y axis under the
    forward-+X/up-+Z convention); the default 0.0 keeps the existing
    pure-proprioception checkpoints unchanged.
    """
    if actuator_type not in _ACTUATOR_TYPES:
        raise ValueError(
            "Only actuator_type='delayed_torque_speed' (explicit PD, hard TN) "
            "and 'implicit' are supported; got "
            f"{actuator_type!r}."
        )
    if actuator_type == "implicit":
        # Implicit PD is an idealization without a command queue.
        min_delay = 0
        max_delay = 0

    from mimic_baseline.robots.rx_p2 import (
        build_rx_p2_action_scale,
        make_rx_p2_articulation_cfg,
    )
    from mimic_baseline.robots.rx_p2_head import (
        NeckControlCfg,
        add_rx_p2_head,
    )

    robot_cfg = make_rx_p2_articulation_cfg(
        actuator_type=actuator_type,
        min_delay=min_delay,
        max_delay=max_delay,
    )
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    for articulation_props in robot_cfg.spawn.articulation_props:
        if isinstance(articulation_props, PhysxArticulationCfg):
            articulation_props.enabled_self_collisions = bool(self_collisions)
        elif isinstance(articulation_props, NewtonArticulationCfg):
            articulation_props.self_collision_enabled = bool(self_collisions)

    action_scale_by_pattern = build_rx_p2_action_scale(robot_cfg.actuators)
    action_scale = []
    for joint_name in _ACTION_JOINT_NAMES:
        matches = [
            value
            for pattern, value in action_scale_by_pattern.items()
            if re.fullmatch(pattern, joint_name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one RX-P2 action scale for {joint_name}, "
                f"found {matches}."
            )
        action_scale.append(float(matches[0]))

    add_rx_p2_head(
        robot_cfg,
        NeckControlCfg(target_positions=(float(neck_yaw), float(neck_pitch))),
    )
    if usd_path is not None:
        # Offline conversion keeps the physics asset and the --urdf FK model
        # explicit, without changing the 27-action/independent-neck interface.
        robot_cfg.spawn.usd_path = str(usd_path)
    return robot_cfg, action_scale


__all__ = ["make_robot_cfg"]
