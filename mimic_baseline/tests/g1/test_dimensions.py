from types import SimpleNamespace
import torch
from mimic_baseline.tasks.supermimic.networks import WorldModel, Policy
from mimic_baseline.tasks.supermimic.buffer import ReplayBuffer
from mimic_baseline.tasks.supermimic.mdp_legacy import MDP
from mimic_baseline.tasks.supermimic.reference import ReferenceState, ReferenceBatch, ReferenceExpiry


def test_g1_network_and_replay_widths_preserve_rxp2_defaults():
    for actions in (27, 29):
        policy = Policy(64 + actions, 15 + 2 * actions, hist=1, hidden=(16,), action_dim=actions)
        model = WorldModel(hidden=(16,), action_dim=actions)
        replay = ReplayBuffer(8, 2, "cpu", action_dim=actions)
        assert model.in_dim == 71 + 2 * actions + 33
        assert policy.decoder[-1].out_features == actions
        assert replay.data["a"].shape == (8, 2, actions)
    assert Policy(91, 69, hist=1, hidden=(16,)).decoder[-1].out_features == 27


def test_g1_goal_retains_both_final_wrist_joints_and_gradients():
    q = torch.arange(29.0).reshape(1, 29)
    R = torch.eye(3).unsqueeze(0)
    p = torch.zeros(1, 3)
    now = ReferenceState(p, R, p, p, q, q, p, p)
    future_q = q.unsqueeze(1).clone().requires_grad_()
    future = ReferenceState(p[:, None], R[:, None], p[:, None], p[:, None], future_q, future_q, p[:, None], p[:, None])
    expiry = ReferenceExpiry(torch.tensor([False]), torch.tensor([False]), torch.tensor([2]))
    batch = ReferenceBatch(now, future, torch.tensor([True]), expiry, action_dim=29)
    goal = batch.goal()
    assert goal.shape == (1, 73)
    assert goal[0, 43] == 28
    goal.sum().backward()
    assert future_q.grad[0, 0, -1] != 0
    legacy = ReferenceBatch(now, future, torch.tensor([True]), expiry)
    assert legacy.goal().shape == (1, 69)


def test_g1_mdp_observation_has_29_previous_actions():
    kin = SimpleNamespace(action_dim=29, q_lo=-torch.ones(29), q_hi=torch.ones(29))
    motion = SimpleNamespace(kin=kin)
    mdp = MDP(kin, motion, torch.zeros(29))
    p = torch.zeros(2, 3)
    q = torch.zeros(2, 29)
    R = torch.eye(3).expand(2, 3, 3)
    observation = mdp.obs((p, R, p, p, q, q), q, torch.zeros(2, 33))
    assert observation.shape == (2, 93)
    assert mdp.obs_noise.shape == (93,)


def test_observation_noise_uses_robot_device_not_global_default_device():
    kin = SimpleNamespace(action_dim=29, q_lo=-torch.ones(29), q_hi=torch.ones(29))
    motion = SimpleNamespace(kin=kin)
    default_q = torch.zeros(29)
    with torch.device("meta"):
        mdp = MDP(kin, motion, default_q)
    assert mdp.obs_noise.device.type == "cpu"


def test_checkpoint_provenance_uses_robot_dimensions():
    from mimic_baseline.tasks.supermimic.recovery import recovery_provenance

    old = recovery_provenance(False)
    assert old['action_dim'] == 27 and old['actor_observation_dim'] == 91
    g1 = recovery_provenance(False, action_dim=29)
    assert g1['action_dim'] == 29 and g1['actor_observation_dim'] == 93
