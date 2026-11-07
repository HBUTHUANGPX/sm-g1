"""SuperMimic mdp package — re-exported for ManagerBasedRLEnv config use.

Legacy constants (ACT_DIM, NUM_J, STATE_DIM, etc.) and the MDP class are
re-exported from mdp_legacy so existing tests and scripts continue to work
after mdp.py was renamed.

Manager-based action/observation/reward/termination terms are importable via
  from mimic_baseline.tasks.supermimic.mdp.actions import ResidualJointPositionAction
  etc.
They are NOT re-exported at package level here to avoid importing isaaclab
modules before SimulationApp is initialized.
"""

# ── legacy constants re-exported for backward compatibility ───────────────────
# Scripts that do `from mimic_baseline.tasks.supermimic.mdp import ACT_DIM`
# continue to work unchanged.  These imports have NO isaaclab dependency.
from ..mdp_legacy import (
    ACT_DIM,
    NUM_J,
    STATE_DIM,
    OBS_DIM,
    XI_DIM,
    INV_DIM,
    CRITIC_EXTRA,
    MAX_STEP_REWARD,
    OBS_NOISE,
    RewardCfg,
    MDP,
    flatten_state,
    unflatten_state,
    invariant_features,
    normalize_xi,
    relative_root_orientation,
    RELATIVE_ORIENTATION_DIM,
    RELATIVE_ORIENTATION_NOISE,
)

__all__ = [
    # legacy re-exports (no isaaclab dependency)
    "ACT_DIM",
    "NUM_J",
    "STATE_DIM",
    "OBS_DIM",
    "XI_DIM",
    "INV_DIM",
    "CRITIC_EXTRA",
    "MAX_STEP_REWARD",
    "OBS_NOISE",
    "RewardCfg",
    "MDP",
    "flatten_state",
    "unflatten_state",
    "invariant_features",
    "normalize_xi",
    "relative_root_orientation",
    "RELATIVE_ORIENTATION_DIM",
    "RELATIVE_ORIENTATION_NOISE",
    # manager-based terms (importable from submodules after SimulationApp init)
    "ResidualJointPositionAction",
    "joint_pos_residual",
    "joint_vel",
    "body_ang_vel",
    "projected_gravity",
    "motion_goal",
    "last_action",
    "body_position_tracking",
    "body_orientation_tracking",
    "body_linvel_tracking",
    "body_angvel_tracking",
    "root_position_tracking",
    "root_orientation_tracking",
    "action_rate_penalty",
    "joint_limit_penalty",
    "height_termination",
    "tilt_termination",
    "motion_end",
    "time_out",
    "randomize_physics_parameters",
    "update_motion_bin_returns",
]


def __getattr__(name: str):
    """Lazy-import manager-based terms to avoid pxr loading before SimulationApp."""
    _actions = {
        "ResidualJointPositionAction": (".actions", "ResidualJointPositionAction"),
    }
    _observations = {
        "joint_pos_residual": (".observations", "joint_pos_residual"),
        "joint_vel": (".observations", "joint_vel"),
        "body_ang_vel": (".observations", "body_ang_vel"),
        "projected_gravity": (".observations", "projected_gravity"),
        "motion_goal": (".observations", "motion_goal"),
        "last_action": (".observations", "last_action"),
    }
    _rewards = {
        "body_position_tracking": (".rewards", "body_position_tracking"),
        "body_orientation_tracking": (".rewards", "body_orientation_tracking"),
        "body_linvel_tracking": (".rewards", "body_linvel_tracking"),
        "body_angvel_tracking": (".rewards", "body_angvel_tracking"),
        "root_position_tracking": (".rewards", "root_position_tracking"),
        "root_orientation_tracking": (".rewards", "root_orientation_tracking"),
        "action_rate_penalty": (".rewards", "action_rate_penalty"),
        "joint_limit_penalty": (".rewards", "joint_limit_penalty"),
    }
    _terminations = {
        "height_termination": (".terminations", "height_termination"),
        "tilt_termination": (".terminations", "tilt_termination"),
        "motion_end": (".terminations", "motion_end"),
        "time_out": (".terminations", "time_out"),
    }
    _events = {
        "randomize_physics_parameters": (".events", "randomize_physics_parameters"),
    }
    _curriculum = {
        "update_motion_bin_returns": (".curriculum", "update_motion_bin_returns"),
    }
    all_lazy = {**_actions, **_observations, **_rewards, **_terminations, **_events, **_curriculum}
    if name in all_lazy:
        import importlib
        module_suffix, attr = all_lazy[name]
        module = importlib.import_module(module_suffix, package=__name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
