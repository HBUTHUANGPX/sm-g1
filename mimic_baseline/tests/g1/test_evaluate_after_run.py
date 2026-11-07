"""Post-run evaluation must never race training or its existing evaluation."""

import hashlib
import json
import pytest
from mimic_baseline.tasks.supermimic.cli import evaluate_after_run as queue


def test_waits_until_complete_then_verifies_and_evaluates(tmp_path):
    run = tmp_path / "run"
    (run / "train").mkdir(parents=True)
    checkpoint = run / "train/model_20000.pt"
    checkpoint.write_bytes(b"checkpoint")
    status = dict(
        phase="training",
        source_revision="a" * 40,
        target_iteration=20000,
        checkpoint_iteration=20000,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    path = run / "status.json"
    path.write_text(json.dumps(status))
    calls = []

    def sleep(seconds):
        calls.append(("sleep", seconds))
        assert not any(c[0] == "evaluate" for c in calls)
        status["phase"] = "evaluating" if status["phase"] == "training" else "complete"
        path.write_text(json.dumps(status))

    def evaluate(*args, **kwargs):
        calls.append(("evaluate", args, kwargs))
        return {"done": True}

    result = queue.run_queue(
        run,
        "a" * 40,
        "b" * 40,
        20000,
        tmp_path / "data",
        tmp_path / "queue",
        sleep=sleep,
        verify_source=lambda rev: calls.append(("verify", rev)),
        load_checkpoint=lambda p: {"iteration": 20000, "format": "supermimic-g1-v1"},
        evaluate=evaluate,
    )
    assert result["phase"] == "complete"
    assert [c[0] for c in calls].count("sleep") == 2
    assert calls[-1][2] == {"profile": "both"}
    assert calls[-1][1] == (
        checkpoint.resolve(),
        (tmp_path / "data").resolve(),
        (tmp_path / "queue/evaluation").resolve(),
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"phase": "training_failed"}, "failed"),
        ({"source_revision": "c" * 40}, "revision"),
        ({"target_iteration": 1}, "iteration"),
        ({"checkpoint_iteration": 1}, "iteration"),
        ({"checkpoint_sha256": "0" * 64}, "digest"),
    ],
)
def test_rejects_invalid_completed_run(tmp_path, change, match):
    run = tmp_path / "run"
    (run / "train").mkdir(parents=True)
    checkpoint = run / "train/model_20000.pt"
    checkpoint.write_bytes(b"x")
    status = dict(
        phase="complete",
        source_revision="a" * 40,
        target_iteration=20000,
        checkpoint_iteration=20000,
        checkpoint_sha256=hashlib.sha256(b"x").hexdigest(),
    )
    status.update(change)
    (run / "status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match=match):
        queue.run_queue(
            run,
            "a" * 40,
            "b" * 40,
            20000,
            tmp_path / "data",
            tmp_path / "queue",
            verify_source=lambda rev: None,
            evaluate=lambda *a, **k: pytest.fail("evaluation must not start"),
        )
    assert json.loads((tmp_path / "queue/status.json").read_text())["phase"] == "failed"


def test_timeout_without_importing_evaluation(tmp_path):
    ticks = iter([0, 2])
    with pytest.raises(TimeoutError):
        queue.run_queue(
            tmp_path / "run",
            "a" * 40,
            "b" * 40,
            20000,
            tmp_path / "data",
            tmp_path / "queue",
            timeout=1,
            clock=lambda: next(ticks),
            verify_source=lambda rev: None,
            evaluate=lambda *a, **k: pytest.fail("evaluation must not start"),
        )


def test_rejects_checkpoint_payload_iteration(tmp_path):
    run = tmp_path / "run"
    (run / "train").mkdir(parents=True)
    (run / "train/model_20000.pt").write_bytes(b"x")
    (run / "status.json").write_text(
        json.dumps(
            dict(
                phase="complete",
                source_revision="a" * 40,
                target_iteration=20000,
                checkpoint_iteration=20000,
                checkpoint_sha256=hashlib.sha256(b"x").hexdigest(),
            )
        )
    )
    with pytest.raises(ValueError, match="payload"):
        queue.run_queue(
            run,
            "a" * 40,
            "b" * 40,
            20000,
            tmp_path / "data",
            tmp_path / "queue",
            verify_source=lambda rev: None,
            load_checkpoint=lambda p: {"iteration": 19999},
            evaluate=lambda *a, **k: pytest.fail("evaluation must not start"),
        )


def test_release_failure_is_recorded_before_wait(tmp_path):
    def reject(revision):
        raise ValueError("release modified")

    with pytest.raises(ValueError, match="release modified"):
        queue.run_queue(
            tmp_path / "run",
            "a" * 40,
            "b" * 40,
            20000,
            tmp_path / "data",
            tmp_path / "queue",
            verify_source=reject,
            sleep=lambda s: pytest.fail("must fail before waiting"),
        )
    assert json.loads((tmp_path / "queue/status.json").read_text())["phase"] == "failed"


def test_waiting_imports_no_simulator_or_torch():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    code = "import sys; sys.path.insert(0, 'mimic_baseline/source/mimic_baseline'); from mimic_baseline.tasks.supermimic.cli import evaluate_after_run; assert not any(x in sys.modules for x in ['torch', 'mujoco', 'isaaclab', 'isaacsim'])"
    subprocess.run([sys.executable, "-c", code], cwd=repo, check=True)
