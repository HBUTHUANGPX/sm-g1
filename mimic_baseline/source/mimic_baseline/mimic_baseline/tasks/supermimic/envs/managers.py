"""Original ManagerBased managers responsibilities."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import RewardManager, TerminationManager


class SuperMimicRewardManager(RewardManager):
    """Capture the physical transition before the base environment's autoreset."""

    def compute(self, dt: float) -> torch.Tensor:
        reward = super().compute(dt)
        self._env._finish_transition(reward)
        return reward


class SuperMimicTerminationManager(TerminationManager):
    """Keep native raw terms; only accepted tracking draws terminate an episode."""

    def __init__(self, cfg, env):
        from mimic_baseline.tasks.supermimic.core.recovery import TRACKING_GROUPS

        super().__init__(cfg, env)
        self._tracking_indices = [
            self._term_name_to_term_idx[name] for name in TRACKING_GROUPS
        ]
        self._tracking_ids = torch.tensor(self._tracking_indices, device=self.device)
        self.accepted_tracking = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.bool
        )

    def compute(self):
        from mimic_baseline.tasks.supermimic.core.recovery import (
            sample_tracking_terminations,
        )

        self._truncated_buf.zero_()
        self._terminated_buf.zero_()
        for index, cfg in enumerate(self._term_cfgs):
            value = cfg.func(self._env, **cfg.params)
            self._term_dones[:, index] = value
            if cfg.time_out:
                self._truncated_buf |= value
            elif index not in self._tracking_indices:
                self._terminated_buf |= value
        # Lab increments the episode counter before this call. Errors already
        # use the physical post-state and command.next_reference_batch().
        raw = self._term_dones[:, self._tracking_ids]
        if self._env.cfg.eval_mode:
            self.accepted_tracking.copy_(raw)
        else:
            self.accepted_tracking.copy_(
                sample_tracking_terminations(
                    raw,
                    self._env.recovery_role,
                    self._env.episode_length_buf - 1,
                    dt=self._env.step_dt,
                )
            )
        self._terminated_buf |= self.accepted_tracking.any(-1)
        accepted = self._term_dones.clone()
        accepted[:, self._tracking_ids] = self.accepted_tracking
        self._last_episode_dones.copy_(
            torch.where(self.dones[:, None], accepted, self._last_episode_dones)
        )
        return self.dones
