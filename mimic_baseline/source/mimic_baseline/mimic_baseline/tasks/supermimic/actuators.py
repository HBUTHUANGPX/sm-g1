"""SuperMimic actuator handling for zero-time Manager reset writes."""

from isaaclab.actuators import IdealPDActuator

from mimic_baseline.robots.torque_speed_actuator import DelayedTorqueSpeedPDActuator


class SuperMimicDelayedPDActuator(DelayedTorqueSpeedPDActuator):
    """Advance command history only when the environment advances physics.

    ManagerBasedEnv.reset writes targets to synchronize the scene, without
    integrating physics. Processing that write as a new delay tick inserts a
    default target before the policy's first action and advances non-reset rows.
    The reset write still computes PD/TN, but leaves all delay buffers intact.
    """

    reset_sync_only = False

    def compute(self, control_action, joint_pos, joint_vel):
        if self.reset_sync_only:
            self._joint_velocity = joint_vel
            return IdealPDActuator.compute(self, control_action, joint_pos, joint_vel)
        return super().compute(control_action, joint_pos, joint_vel)


from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils.configclass import configclass
from mimic_baseline.robots.actuator_delay_buffer import ActuatorDelayBuffer


class G1DelayedPDActuator(DelayedPDActuator):
    """Original delay/reset contract with static effort limits, without a TN curve."""

    reset_sync_only = False

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        for name in ("positions", "velocities", "efforts"):
            setattr(self, f"{name}_delay_buffer", ActuatorDelayBuffer(cfg.max_delay, self._num_envs, self._device))

        # Isaac Lab GA reset iterates a cached tuple, not the public fields.
        self._delay_buffers = (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer)

    def compute(self, control_action, joint_pos, joint_vel):
        if self.reset_sync_only:
            return IdealPDActuator.compute(self, control_action, joint_pos, joint_vel)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class G1DelayedPDActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = G1DelayedPDActuator
    min_delay: int = 0
    max_delay: int = 4
