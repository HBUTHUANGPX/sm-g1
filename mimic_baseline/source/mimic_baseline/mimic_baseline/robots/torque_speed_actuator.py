"""IsaacLab actuator adapter for a precomputed torque-speed lookup table."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, IdealPDActuator
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions

from .torque_speed import TorqueSpeedLimiter, normalized_torque_speed_excess
from .actuator_delay_buffer import ActuatorDelayBuffer


class DelayedTorqueSpeedPDActuator(DelayedPDActuator):
    """Delayed explicit PD actuator with a measured torque-speed envelope."""

    cfg: DelayedTorqueSpeedPDActuatorCfg

    def __init__(self, cfg: DelayedTorqueSpeedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # Keep IsaacLab's delay sampling/reset lifecycle, replacing only storage.
        # Its general-purpose buffer synchronizes and rolls history every step.
        self.positions_delay_buffer = ActuatorDelayBuffer(
            cfg.max_delay, self._num_envs, self._device, delay_step_size=cfg.delay_step_size,
        )
        self.velocities_delay_buffer = ActuatorDelayBuffer(
            cfg.max_delay, self._num_envs, self._device, delay_step_size=cfg.delay_step_size,
        )
        self.efforts_delay_buffer = ActuatorDelayBuffer(
            cfg.max_delay, self._num_envs, self._device, delay_step_size=cfg.delay_step_size,
        )
        self._torque_speed_limiter = TorqueSpeedLimiter(
            speed_step_rpm=cfg.speed_step_rpm,
            torque_limits_nm=cfg.torque_limits_nm,
            dtype=self.computed_effort.dtype,
            device=self._device,
        )
        self._joint_velocity = self.computed_effort
        if not cfg.hard_tn_limit:
            self.tn_excess = torch.zeros_like(self.computed_effort)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_velocity = joint_vel
        positions = self.positions_delay_buffer
        control_action.joint_positions = positions.compute(control_action.joint_positions)
        # Native reset assigns identical lags to all three setpoints. Reuse the
        # position history index, then run the unchanged PD and TN computation.
        control_action.joint_velocities = self.velocities_delay_buffer.compute(
            control_action.joint_velocities, read_indices=positions._read_indices,
        )
        control_action.joint_efforts = self.efforts_delay_buffer.compute(
            control_action.joint_efforts, read_indices=positions._read_indices,
        )
        return IdealPDActuator.compute(self, control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        if not self.cfg.hard_tn_limit:
            # Preserve the static motor limit, but let reward discourage TN
            # violations. Cache the cost at the same pre-integration velocity
            # as the applied torque; the control-step reward reads the last substep.
            applied = super()._clip_effort(effort)
            self.tn_excess = normalized_torque_speed_excess(
                applied, self._torque_speed_limiter.lookup(self._joint_velocity),
                self.cfg.torque_limits_nm[0],
            )
            return applied
        return self._torque_speed_limiter.clip(
            effort,
            self._joint_velocity,
        )


@configclass
class DelayedTorqueSpeedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`DelayedTorqueSpeedPDActuator`."""

    class_type: type = DelayedTorqueSpeedPDActuator

    hard_tn_limit: bool = True
    """False uses the static effort limit and exposes bounded tn_excess for reward."""

    delay_step_size: int = 1
    """Physics steps per min/max delay tick; decimation preserves policy-period delays."""

    speed_step_rpm: float = MISSING
    """Uniform lookup-table speed step in rpm."""

    torque_limits_nm: tuple[float, ...] = MISSING
    """Maximum absolute torque at each lookup-table speed."""
