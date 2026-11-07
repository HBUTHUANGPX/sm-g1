"""Vectorized independent joint delay, preserving the original buffer arithmetic."""

import torch
from mimic_baseline.robots.actuator_delay_buffer import ActuatorDelayBuffer


def sample_joint_delays(draw_order, joint_order, count, low, high, device):
    # One original-shaped RNG call per joint, in original actuator insertion order.
    draws = {
        name: torch.randint(low, high + 1, (count,), dtype=torch.int, device=device)
        for name in draw_order
    }
    return torch.stack([draws[name] for name in joint_order], dim=1)


class JointDelayBuffer:
    def __init__(self, max_delay, num_envs, num_joints, device):
        self.shape = (num_envs, num_joints)
        self.buffer = ActuatorDelayBuffer(max_delay, num_envs * num_joints, device)
        self.ids = torch.arange(num_envs * num_joints, device=device).reshape(
            self.shape
        )

    @property
    def time_lags(self):
        return self.buffer.time_lags.view(self.shape)

    def set_time_lag(self, lags, env_ids=None):
        self.time_lags[slice(None) if env_ids is None else env_ids] = lags.to(self.time_lags)

    def reset(self, env_ids=None):
        self.buffer.reset(None if env_ids is None else self.ids[env_ids].reshape(-1))

    def compute(self, value):
        return self.buffer.compute(value.reshape(-1, 1)).reshape(self.shape)
