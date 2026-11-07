"""Evaluation-only batching of identical per-joint delayed PD computations."""

from isaaclab.actuators import IdealPDActuator
from .actuators import G1DelayedPDActuator
from ..evaluation.joint_delay import JointDelayBuffer, sample_joint_delays


class GroupedEvaluationPD(G1DelayedPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        for name in ("positions", "velocities", "efforts"):
            setattr(
                self,
                f"{name}_delay_buffer",
                JointDelayBuffer(
                    cfg.max_delay, self._num_envs, len(self._joint_names), self._device
                ),
            )
        self._delay_buffers = (
            self.positions_delay_buffer,
            self.velocities_delay_buffer,
            self.efforts_delay_buffer,
        )

    def reset(self, env_ids):
        IdealPDActuator.reset(self, env_ids)
        ids = slice(None) if env_ids is None else env_ids
        count = len(range(self._num_envs)[ids]) if isinstance(ids, slice) else len(ids)
        lags = sample_joint_delays(
            self.cfg.joint_names_expr,
            self._joint_names,
            count,
            self.cfg.min_delay,
            self.cfg.max_delay,
            self._device,
        )
        for buffer in self._delay_buffers:
            buffer.set_time_lag(lags, ids)
            buffer.reset(ids)


def group_evaluation_actuators(cfg):
    if not cfg.eval_mode:
        raise ValueError("Grouped actuators are evaluation-only")
    if cfg.actuator_type != "delayed_pd":
        return
    old = cfg.scene.robot.actuators
    if not old or any(
        a.class_type is not G1DelayedPDActuator or a.joint_names_expr != [name]
        for name, a in old.items()
    ):
        raise ValueError("Expected one original G1 delayed PD actuator per named joint")
    first = next(iter(old.values()))
    if any(
        (a.min_delay, a.max_delay) != (first.min_delay, first.max_delay)
        for a in old.values()
    ):
        raise ValueError("Inconsistent delay bounds")
    grouped = first.copy()
    grouped.class_type = GroupedEvaluationPD
    grouped.joint_names_expr = list(old)
    for field in (
        "stiffness",
        "damping",
        "effort_limit",
        "velocity_limit",
        "effort_limit_sim",
        "velocity_limit_sim",
        "actuator_effort_limit",
        "actuator_velocity_limit",
        "joint_effort_limit",
        "joint_velocity_limit",
        "armature",
        "friction",
        "static_friction",
        "dynamic_friction",
        "viscous_friction",
    ):
        if not hasattr(first, field):
            continue
        values = {name: getattr(a, field) for name, a in old.items()}
        if all(v is None for v in values.values()):
            setattr(grouped, field, None)
        elif all(isinstance(v, (float, int)) for v in values.values()):
            setattr(grouped, field, values)
        else:
            raise ValueError(f"Unsupported grouped parameter: {field}")
    cfg.scene.robot.actuators = {"g1_evaluation": grouped}
