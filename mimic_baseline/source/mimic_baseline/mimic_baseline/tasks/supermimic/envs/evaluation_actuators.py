"""Compatibility entry point for the verified evaluation actuator batching."""

from .grouped_actuators import GroupedG1DelayedPD, group_g1_actuators

GroupedEvaluationPD = GroupedG1DelayedPD


def group_evaluation_actuators(cfg, *, allow_training=False):
    """Retain evaluation callers and the bounded profiling opt-in."""
    if not cfg.eval_mode and not allow_training:
        raise ValueError("Grouped actuators are evaluation-only")
    return group_g1_actuators(cfg, group_name="g1_evaluation")
