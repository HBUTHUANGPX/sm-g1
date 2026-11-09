"""Configuration contract; Isaac Lab runtime parity lives in native validation."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def grouped_module(monkeypatch):
    # Only unavailable simulator imports are stubbed. Configuration conversion is real.
    package = '_grouped_contract.envs'
    root = ModuleType(package)
    root.__path__ = []
    monkeypatch.setitem(sys.modules, package, root)
    native = ModuleType('isaaclab.actuators')
    native.IdealPDActuator = type('IdealPDActuator', (), {})
    monkeypatch.setitem(sys.modules, 'isaaclab.actuators', native)
    actuators = ModuleType(package + '.actuators')
    actuators.G1DelayedPDActuator = type('G1DelayedPDActuator', (), {})
    monkeypatch.setitem(sys.modules, package + '.actuators', actuators)
    path = Path(__file__).parents[2] / 'source/mimic_baseline/mimic_baseline/tasks/supermimic/envs/grouped_actuators.py'
    assert path.exists(), 'Shared training actuator implementation is missing'
    # Real delay implementation is independent of the simulation runtime.
    from mimic_baseline.tasks.supermimic.evaluation import joint_delay
    monkeypatch.setitem(sys.modules, '_grouped_contract.evaluation', ModuleType('_grouped_contract.evaluation'))
    monkeypatch.setitem(sys.modules, '_grouped_contract.evaluation.joint_delay', joint_delay)
    spec = importlib.util.spec_from_file_location(package + '.grouped_actuators', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, spec.name, module)
    return module, actuators.G1DelayedPDActuator


class Config(SimpleNamespace):
    def copy(self):
        return copy.deepcopy(self)


def make_config(native):
    old = {name: Config(class_type=native, joint_names_expr=[name], min_delay=0,
                        max_delay=4, stiffness=20.0 + k, damping=1.0 + k,
                        effort_limit=30.0 + k, velocity_limit=10.0 + k,
                        armature=0.01 + k, friction=None)
           for k, name in enumerate(['hip', 'ankle', 'knee'])}
    return SimpleNamespace(eval_mode=False, actuator_type='delayed_pd',
                           scene=SimpleNamespace(robot=SimpleNamespace(actuators=old)))


def test_training_grouping_preserves_named_parameters_without_mutating_source(grouped_module):
    module, native = grouped_module
    cfg = make_config(native)
    old = cfg.scene.robot.actuators
    module.group_g1_actuators(cfg)
    grouped = cfg.scene.robot.actuators['g1_grouped']
    assert grouped.class_type is module.GroupedG1DelayedPD
    assert grouped.joint_names_expr == list(old)
    for field in ('stiffness', 'damping', 'effort_limit', 'velocity_limit', 'armature'):
        assert getattr(grouped, field) == {name: getattr(value, field) for name, value in old.items()}
    assert grouped.friction is None
    assert all(value.class_type is native for value in old.values())


def test_grouping_rejects_mixed_delays_atomically(grouped_module):
    module, native = grouped_module
    cfg = make_config(native)
    old = cfg.scene.robot.actuators
    old['ankle'].max_delay = 3
    with pytest.raises(ValueError, match='delay bounds'):
        module.group_g1_actuators(cfg)
    assert cfg.scene.robot.actuators is old


def test_evaluation_entrypoint_retains_guard_alias_and_group_key(grouped_module):
    module, native = grouped_module
    path = Path(module.__file__).with_name('evaluation_actuators.py')
    spec = importlib.util.spec_from_file_location('_grouped_contract.envs.evaluation_actuators', path)
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    assert compat.GroupedEvaluationPD is module.GroupedG1DelayedPD
    cfg = make_config(native)
    with pytest.raises(ValueError, match='evaluation-only'):
        compat.group_evaluation_actuators(cfg)
    cfg.eval_mode = True
    compat.group_evaluation_actuators(cfg)
    assert list(cfg.scene.robot.actuators) == ['g1_evaluation']
    cfg = make_config(native)
    compat.group_evaluation_actuators(cfg, allow_training=True)
    assert list(cfg.scene.robot.actuators) == ['g1_evaluation']
