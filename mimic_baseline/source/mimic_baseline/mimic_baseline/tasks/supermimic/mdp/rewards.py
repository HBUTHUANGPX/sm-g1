"""Manager reward terms sharing the analytic SuperMimic MDP's FK/errors."""

from __future__ import annotations

import torch


def body_position_tracking(env, command_name="motion", std=0.3):
    return torch.exp(-env.reward_terms()["e_body_pos"] / std**2)


def body_orientation_tracking(env, command_name="motion", std=0.4):
    return torch.exp(-env.reward_terms()["e_body_ori"] / std**2)


def body_linvel_tracking(env, command_name="motion", std=1.0):
    return torch.exp(-env.reward_terms()["e_body_linvel"] / std**2)


def body_angvel_tracking(env, command_name="motion", std=3.14):
    return torch.exp(-env.reward_terms()["e_body_angvel"] / std**2)


def root_position_tracking(env, command_name="motion", std=0.3):
    return torch.exp(-env.reward_terms()["e_root_pos"] / std**2)


def root_orientation_tracking(env, command_name="motion", std=0.4):
    return torch.exp(-env.reward_terms()["e_root_ori"] / std**2)


def action_rate_penalty(env):
    return env.reward_terms()["action_rate"]


def joint_limit_penalty(env):
    return env.reward_terms()["joint_limit"]
