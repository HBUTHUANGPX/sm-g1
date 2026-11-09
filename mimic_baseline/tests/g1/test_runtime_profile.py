import subprocess
import sys
from pathlib import Path
import pytest
from mimic_baseline.tasks.supermimic.cli.profile_runtime import audit, portable_function


def test_profile_rejects_unbounded_training_before_creating_outputs(tmp_path):
    with pytest.raises(ValueError, match="Bounded"):
        audit(tmp_path / "missing", tmp_path, tmp_path / "output", iterations=3000)
    assert not (tmp_path / "output").exists()


def test_profile_function_paths_do_not_publish_machine_locations():
    assert (
        portable_function(
            "/private/home/source/tasks/supermimic/learning/policy.py", 12, "update"
        )
        == "learning/policy.py:12:update"
    )
    assert (
        portable_function(
            "C:\\Users\\Private\\site-packages\\torch\\autograd.py", 1, "backward"
        )
        == "torch/autograd.py:1:backward"
    )


def test_profile_cli_help_without_simulator():
    root = Path(__file__).resolve().parents[3]
    assert "Bounded" in subprocess.check_output(
        [sys.executable, str(root / "scripts/profile_g1_runtime.py"), "--help"],
        text=True,
    )


def test_state_cache_validation_is_evaluation_only(tmp_path):
    import pytest
    from mimic_baseline.tasks.supermimic.cli.profile_runtime import audit
    with pytest.raises(ValueError, match="requires evaluation"):
        audit(tmp_path / "missing.pt", tmp_path, tmp_path / "out",
              target="training", verify_state_cache=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("warmup", [-1, 5, 6])
def test_profile_rejects_invalid_measurement_window(tmp_path, warmup):
    with pytest.raises(ValueError, match="warmup"):
        audit(tmp_path / "missing", tmp_path, tmp_path / "output",
              iterations=5, warmup_iterations=warmup)
    assert not (tmp_path / "output").exists()


def test_training_window_keeps_original_calls_and_measures_final_save():
    from mimic_baseline.tasks.supermimic.cli import profile_runtime as module
    assert hasattr(module, 'run_training_window'), 'Training window hook is missing'
    events = []
    now = [0.0]
    class Trainer:
        iteration = 100
        def collect(self, count):
            events.append(('collect', count))
            now[0] += count
        def _set_lr(self, iteration):
            events.append(('lr', iteration))
        def train(self):
            self.collect(10)
            for i in range(100, 103):
                self._set_lr(i)
                self.collect(2)
                self.iteration = i + 1
            now[0] += 7  # Original final checkpoint serialization belongs to last iteration.
    trainer = Trainer()
    timing = {}
    module.run_training_window(
        trainer, Trainer.train, {}, warmup_iterations=1, timing=timing,
        start=lambda: events.append(('start', trainer.iteration)),
        finish=lambda: events.append(('finish', trainer.iteration)),
        synchronize=lambda: None, clock=lambda: now[0],
    )
    assert timing['refill_seconds'] == 10
    assert timing['iteration_seconds'] == [2, 9]
    assert timing['measured_training_iterations'] == 2
    assert ('start', 101) in events
    assert ('finish', 103) in events
    assert [x for x in events if x[0] == 'lr'] == [('lr', 100), ('lr', 101), ('lr', 102)]
    assert '_set_lr' not in trainer.__dict__
    assert 'collect' not in trainer.__dict__


def test_training_window_restores_instance_overrides_after_failure():
    from types import SimpleNamespace
    from mimic_baseline.tasks.supermimic.cli.profile_runtime import run_training_window
    collect = lambda count: None
    set_lr = lambda iteration: None
    trainer = SimpleNamespace(iteration=8, collect=collect, _set_lr=set_lr)
    finished = []
    timing = {}
    def train(instance):
        instance.collect(4)
        instance._set_lr(8)
        raise RuntimeError('training failure')
    with pytest.raises(RuntimeError, match='training failure'):
        run_training_window(trainer, train, {}, warmup_iterations=0, timing=timing,
                            start=lambda: None, finish=lambda: finished.append(True),
                            synchronize=lambda: None, clock=lambda: 0)
    assert trainer.collect is collect
    assert trainer._set_lr is set_lr
    assert finished == [True]
    assert timing['measured_training_iterations'] == 0


@pytest.mark.parametrize('kwargs', [
    {'target': 'evaluation', 'warmup_iterations': 1},
    {'fk_chunk_size': 0}, {'kinematics': 'unknown'}, {'actuators': 'unknown'},
])
def test_profile_rejects_invalid_execution_before_creating_outputs(tmp_path, kwargs):
    with pytest.raises(ValueError):
        audit(tmp_path / 'missing', tmp_path, tmp_path / 'out', **kwargs)
    assert not (tmp_path / 'out').exists()
