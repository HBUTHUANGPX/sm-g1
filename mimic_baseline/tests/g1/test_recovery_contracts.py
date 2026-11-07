from types import SimpleNamespace
import torch
import pytest
from mimic_baseline.tasks.supermimic.recovery import sample_tracking_terminations, soft_tracking_hazard, remaining_protection
from mimic_baseline.tasks.supermimic.g1_run_config import build_g1_models, restore_run_config


def test_recovery_networks_and_replay_preserve_g1_dimensions():
    from mimic_baseline.tasks.supermimic.buffer import ReplayBuffer
    cfg = SimpleNamespace(wm_target_mode='residual',recovery_enabled=True,hist=10,goal_frames=10,
                          actor_relative_orientation=True,device='cpu')
    wm,policy,critic = build_g1_models(cfg,small_model=True)
    assert wm.in_dim == 162 and wm.out_dim == 73
    assert policy.action_dim == 29 and policy.obs_dim == 93 and policy.goal_dim == 730
    assert critic.in_dim == 915
    replay = ReplayBuffer(8,2,'cpu',action_dim=29,recovery_enabled=True)
    assert replay.data['a'].shape == (8,2,29)
    assert replay.data['recovery_role'].dtype == torch.bool


def test_original_four_group_probabilities_and_protection_boundary():
    torch.manual_seed(82)
    raw = torch.ones(20000,4,dtype=torch.bool)
    accepted = sample_tracking_terminations(raw,torch.zeros(20000,dtype=torch.bool),torch.zeros(20000,dtype=torch.long),dt=.02)
    torch.testing.assert_close(accepted.float().mean(0),torch.tensor([.02/3,.02/3,.04,.04]),atol=.005,rtol=0)
    role = torch.tensor([True,True,False]); age = torch.tensor([199,200,0])
    accepted = sample_tracking_terminations(torch.ones(3,4,dtype=torch.bool),role,age,dt=4.)
    assert accepted.any(-1).tolist() == [False,True,True]
    torch.testing.assert_close(remaining_protection(role,age),torch.tensor([.005,0.,0.]))
    errors = torch.full((3,4),.5,requires_grad=True)
    h = soft_tracking_hazard(errors,role,age)
    assert h[0] == 0 and h[1] > 0
    h.sum().backward()
    assert errors.grad[0].abs().sum() == 0 and errors.grad[1].abs().sum() > 0


def test_resume_refuses_flat_to_recovery_and_delay_changes():
    saved = {'format':'supermimic-g1-v1','config':{'robot_name':'g1','recovery_enabled':False,
             'actuator_type':'implicit','max_delay':0},'provenance':{'backend':'newton'}}
    for change in ({'recovery_enabled':True},{'actuator_type':'delayed_pd'},{'max_delay':4}):
        with pytest.raises(ValueError,match='differs'):
            restore_run_config(saved,change)
