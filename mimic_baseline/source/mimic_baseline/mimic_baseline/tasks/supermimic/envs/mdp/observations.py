"""Manager terms in the same canonical order and units as MDP.obs."""

from __future__ import annotations


def joint_pos_residual(env):
    cmd = env.command_manager.get_term("motion")
    return env.scene["robot"].data.joint_pos.torch[:, cmd.joint_ids] - cmd.default_q - env.xi[:, 1:30]


def joint_vel(env):
    cmd = env.command_manager.get_term("motion")
    return env.scene["robot"].data.joint_vel.torch[:, cmd.joint_ids]


def body_ang_vel(env):
    _, rotation, _, angular_velocity, _, _ = env.state()
    return (rotation.transpose(-1, -2) @ angular_velocity.unsqueeze(-1)).squeeze(-1)


def projected_gravity(env):
    return -env.state()[1][:, 2, :]


def ref_root_rot6d_in_root(env):
    from mimic_baseline.tasks.supermimic.core.mdp import relative_root_orientation

    cmd = env.command_manager.get_term("motion")
    return relative_root_orientation(env.state()[1], cmd.reference_batch().now.R)


def motion_goal(env, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    if env.cfg.goal_mode == "state_relative":
        return cmd.reference_batch(goal_frames=cmd.cfg.goal_frames).goal(
            mode="state_relative", rotation=env.state()[1],
        )
    return cmd.compute_goal(env)


def last_action(env):
    # ActionManager clears selected rows at reset and retains the action that
    # was just applied for the next observation.
    return env.action_manager.action
