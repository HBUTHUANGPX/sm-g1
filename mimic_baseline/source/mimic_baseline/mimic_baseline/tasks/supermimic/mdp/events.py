"""Episode physics randomization, before the command writes the reset pose."""

from __future__ import annotations


def randomize_physics_parameters(env, env_ids):
    env.randomization.resample(env_ids)
    # The two neck servos hold a known camera mount, outside the policy action.
    env.xi[env_ids, 1 + getattr(getattr(env, "cfg", None), "action_dim", 27):30] = 0.0
