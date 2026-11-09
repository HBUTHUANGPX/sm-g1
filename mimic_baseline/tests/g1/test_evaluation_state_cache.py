"""CPU lifecycle checks for the real evaluation state method without Isaac Lab."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from mimic_baseline.tasks.supermimic.envs.mdp import observations

ENV_ROOT = (
    Path(__file__).resolve().parents[2]
    / "source/mimic_baseline/mimic_baseline/tasks/supermimic/envs"
)


def methods(filename, class_name, names):
    tree = ast.parse((ENV_ROOT / filename).read_text())
    selected = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return [
        node
        for node in selected.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]


def make_env():
    class Simulator:
        def state(self):
            self.reads += 1
            return tuple(value.clone() for value in self.physical)

        def _reset_idx(self, env_ids):
            self.resets += 1
            self.physical[0][env_ids] = 7

    namespace = {"Simulator": Simulator, "torch": torch}
    classes = [
        ast.ClassDef(
            name="Managed",
            bases=[ast.Name(id="Simulator", ctx=ast.Load())],
            keywords=[],
            body=methods(
                "manager_based.py", "SuperMimicRLEnv", {"post_state", "_reset_idx"}
            ),
            decorator_list=[],
        ),
        ast.ClassDef(
            name="Evaluation",
            bases=[ast.Name(id="Managed", ctx=ast.Load())],
            keywords=[],
            body=methods("evaluation.py", "G1EvaluationEnv", {"state"}),
            decorator_list=[],
        ),
    ]
    exec(  # noqa: S102 - execute only selected methods from trusted project source
        compile(
            ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])),
            "actual_env_methods",
            "exec",
        ),
        namespace,
    )
    env = namespace["Evaluation"]()
    env._stepping = False
    env._post_state = None
    env.cfg = SimpleNamespace(eval_mode=True)
    env.reads = env.resets = 0
    env._reset_sensors = lambda _: None
    env.physical = (
        torch.zeros(2, 3),
        torch.eye(3).repeat(2, 1, 1),
        torch.ones(2, 3),
        torch.ones(2, 3),
        torch.zeros(2, 29),
        torch.zeros(2, 29),
    )
    return env


def test_post_snapshot_is_reused_only_during_current_step():
    env = make_env()
    env._stepping = True
    first = env.post_state()
    assert env.state() is first and env.reads == 1
    # These are the real parent step's boundaries: end stepping, invalidate at
    # next step entry, then physics advances before a new post snapshot is made.
    env._stepping = False
    assert env.state() is not first
    env._post_state = None
    env.physical[0].add_(1)
    env._stepping = True
    second = env.post_state()
    assert second is not first
    assert torch.equal(first[0], torch.zeros(2, 3))
    assert torch.equal(second[0], torch.ones(2, 3))


def test_strict_done_suppresses_reset_but_explicit_reset_reads_new_pose():
    env = make_env()
    env._stepping = True
    snapshot = env.post_state()
    env._reset_idx(torch.tensor([0]))
    assert env.resets == 0 and env.state() is snapshot
    env._stepping = False
    env._reset_idx(torch.tensor([0]))
    assert env.resets == 1
    assert env.state()[0][0, 0] == 7
    assert snapshot[0][0, 0] == 0


def test_cached_state_observation_consumers_do_not_mutate_snapshot():
    env = make_env()
    env._stepping = True
    snapshot = env.post_state()
    original = tuple(value.clone() for value in snapshot)
    angular = observations.body_ang_vel(env)
    gravity = observations.projected_gravity(env)
    assert env.reads == 1
    assert torch.equal(angular, torch.ones(2, 3))
    assert torch.equal(gravity, torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1))
    # Observation manager may scale/clip returned terms in place.
    angular.zero_()
    gravity.zero_()
    for actual, expected in zip(snapshot, original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
