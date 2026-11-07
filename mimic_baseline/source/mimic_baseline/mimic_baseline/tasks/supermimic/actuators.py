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
