import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch

@pytest.mark.parametrize('width',[27,29])
def test_original_action_term_preserves_active_and_fixed_joints(width,monkeypatch):
    class ActionTerm:
        def __init__(self,cfg,env):
            self.cfg=cfg;self._env=env;self.num_envs=2;self.device='cpu'
    stub=ModuleType('isaaclab.managers.action_manager');stub.ActionTerm=ActionTerm
    monkeypatch.setitem(sys.modules,'isaaclab.managers.action_manager',stub)
    path=Path(__file__).resolve().parents[2]/'source/mimic_baseline/mimic_baseline/tasks/supermimic/mdp/actions.py'
    spec=importlib.util.spec_from_file_location('_action_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    command=SimpleNamespace(default_q=torch.zeros(29),joint_ids=torch.arange(29))
    env=SimpleNamespace(scene={'robot':object()},command_manager=SimpleNamespace(get_term=lambda _:command),
        xi=torch.zeros(2,33),action_scale=torch.ones(width))
    term=module.ResidualJointPositionAction(SimpleNamespace(asset_name='robot',action_dim=width),env)
    assert term.action_dim == width
    term.process_actions(torch.ones(2,width))
    assert torch.equal(term.processed_actions,torch.ones(2,width))
    if width==27: assert torch.equal(term._targets[:,-2:],torch.zeros(2,2))
    term.reset(torch.tensor([0]));assert not term.raw_actions[0].any()
    assert term.raw_actions[1].sum() == width
