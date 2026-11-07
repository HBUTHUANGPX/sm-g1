import pytest
import torch
from mimic_baseline.tasks.supermimic.motion import MotionLib


@pytest.mark.parametrize('actions', [27, 29])
def test_legacy_motion_goal_interfaces_follow_active_joints(actions):
    motion = MotionLib.__new__(MotionLib)
    motion.action_dim = actions
    motion.clip_end = torch.tensor([3])
    motion.clip_id = torch.zeros(4, dtype=torch.long)
    motion.root_p = torch.zeros(4, 3)
    motion.root_R = torch.eye(3).repeat(4, 1, 1)
    motion.root_v_local = torch.zeros(4, 3)
    motion.root_w_local = torch.zeros(4, 3)
    motion.q = torch.arange(29.0).repeat(4, 1).requires_grad_()
    motion.qd = torch.zeros(4, 29)
    frames = torch.tensor([0])
    goal = motion.goal(frames, 2)
    legacy = motion.goal_state_relative(frames, 2, torch.eye(3)[None])
    assert goal.shape == (1, (15 + 2 * actions) * 2)
    assert legacy.shape == (1, (6 + 2 * actions) * 2)
    goal.sum().backward()
    assert motion.q.grad[:, actions - 1].sum() == 2
    if actions == 27:
        assert motion.q.grad[:, 27:].sum() == 0
