"""Residual body actions and independent fixed neck PD targets."""

from __future__ import annotations

import torch
from isaaclab.managers.action_manager import ActionTerm


class ResidualJointPositionAction(ActionTerm):
    """q_target = q_default + calibration + scale * action for 27 body joints."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.asset_name]
        self._command = env.command_manager.get_term("motion")
        self._raw_actions = torch.zeros(self.num_envs, cfg.action_dim, device=self.device)
        self._targets = self._command.default_q.expand(self.num_envs, -1).clone()

    @property
    def action_dim(self):
        return self.cfg.action_dim

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._targets[:, :self.action_dim]

    def process_actions(self, actions):
        self._raw_actions.copy_(actions)
        self._targets.copy_(self._command.default_q + self._env.xi[:, 1:30])
        self._targets[:, :self.action_dim].add_(self._env.action_scale * actions)
        self._targets[:, self.action_dim:].copy_(self._command.default_q[self.action_dim:])

    def apply_actions(self):
        # Also write the neck target each substep; an initial pose alone does
        # not configure a PD target in IsaacLab.
        self._robot.set_joint_position_target_index(
            target=self._targets, joint_ids=self._command.joint_ids, env_ids=None,
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._targets[ids] = self._command.default_q + self._env.xi[ids, 1:30]
        self._targets[ids, self.action_dim:] = self._command.default_q[self.action_dim:]
