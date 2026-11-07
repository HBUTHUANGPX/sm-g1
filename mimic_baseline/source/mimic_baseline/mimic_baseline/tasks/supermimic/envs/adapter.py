"""Original ManagerBased adapter responsibilities."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager_based import SuperMimicRLEnv

import math


class SuperMimicTrainingEnv:
    """Expose model-based transitions without changing ManagerBasedRLEnv.step."""

    def __init__(self, env: SuperMimicRLEnv):
        self.env = env
        self.N = env.num_envs
        self.device = env.device

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def reference(self):
        return self.env.reference

    def reference_address(self):
        return self.env.reference_address()

    def reference_batch(self, goal_frames=0, bodies=False):
        return self.env.reference_batch(goal_frames=goal_frames, bodies=bodies)

    @property
    def ep_step(self):
        return self.env.episode_length_buf

    def step(self, action):
        self.env.step(action)
        return self.env.transition

    def policy_observation(self):
        return self.env.obs_buf["supermimic_obs"]

    def policy_relative_orientation(self):
        return self.env.obs_buf["supermimic_relative_orientation"]

    def reset(self, env_ids=None, frames=None):
        if frames is not None:
            return self.env.reset_frames(env_ids, frames)
        return self.env.reset(env_ids=env_ids)

    def close(self):
        self.env.close()
