"""The real-step counterparts of MDP.termination, plus true truncations."""

from __future__ import annotations


def height_termination(env, command_name="motion", threshold=0.20):
    cmd = env.command_manager.get_term(command_name)
    reference = cmd.next_reference_batch().now.p
    return (reference[:, 2] - env.post_state()[0][:, 2]).abs() > threshold


def tilt_termination(env, command_name="motion", threshold=0.8):
    cmd = env.command_manager.get_term(command_name)
    reference = cmd.next_reference_batch().now.R
    return (reference[:, 2, 2] - env.post_state()[1][:, 2, 2]).abs() > threshold


def ankle_termination(env, threshold=0.20):
    return env.tracking_errors()[:, 2] > threshold


def wrist_termination(env, threshold=0.20):
    return env.tracking_errors()[:, 3] > threshold


def motion_end(env, command_name="motion"):
    return env.command_manager.get_term(command_name).at_clip_end()


def time_out(env):
    return env.episode_length_buf >= env.max_episode_length
