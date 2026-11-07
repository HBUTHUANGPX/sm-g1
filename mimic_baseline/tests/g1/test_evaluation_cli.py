import importlib.util
import json
from pathlib import Path
import subprocess
import torch
import pytest

from mimic_baseline.tasks.supermimic.cli import evaluate as cli


def fixture(tmp_path):
    import numpy as np
    from dataclasses import asdict
    from mimic_baseline.tasks.supermimic.learning.config import TrainingConfig
    from mimic_baseline.tasks.supermimic.learning.models import Policy

    data = tmp_path / "data"
    (data / "lafan1_g1").mkdir(parents=True)
    raw = np.zeros((2, 36))
    raw[:, 6] = 1
    for i in range(40):
        np.savetxt(data / "lafan1_g1" / f"clip{i}.csv", raw, delimiter=",")
    for name in (
        "g1_description/g1_29dof_rev_1_0.urdf",
        "g1_usd/g1_29dof/g1_29dof.usda",
        "ground/default_ground_plane.usda",
    ):
        asset = data / name
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text("fixture")
    cfg = TrainingConfig(robot_name="g1", device="cpu", actuator_type="delayed_pd")
    policy = Policy(
        93,
        73 * cfg.goal_frames,
        hist=cfg.hist,
        action_dim=29,
        actor_relative_orientation=cfg.actor_relative_orientation,
        hidden=(64, 64),
    )
    ck = tmp_path / "model_100.pt"
    torch.save(
        {
            "format": "supermimic-g1-v1",
            "iteration": 100,
            "config": asdict(cfg),
            "policy": policy.state_dict(),
            "provenance": {"small_model": True},
        },
        ck,
    )
    return ck, data, tmp_path / "eval"


def test_dry_run_does_not_create_outputs_or_launch(tmp_path, monkeypatch):
    ck, data, out = fixture(tmp_path)
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected simulator")
    )
    r = cli.evaluate(ck, data, out, dry_run=True)
    assert r["preflight_passed"] and len(r["expected_clips"]) == 40 and not out.exists()


@pytest.mark.parametrize("failure", [None, "child", "incomplete", "nonfinite"])
def test_evaluation_outputs_and_failure_gates(tmp_path, monkeypatch, failure):
    ck, data, out = fixture(tmp_path)

    def run(command, **kwargs):
        if failure == "child":
            raise subprocess.CalledProcessError(1, command)
        rows = [
            {"name": f"clip{i}", "completed": i < 3, "normalized_return": 0.5}
            for i in range(39 if failure == "incomplete" else 40)
        ]
        if failure == "nonfinite":
            rows[0]["normalized_return"] = float("nan")
        (out / "perclip.json").write_text(json.dumps({"clips": rows}))

    monkeypatch.setattr(cli.subprocess, "run", run)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            cli.evaluate(ck, data, out)
        assert (out / "failure.json").exists() and not (out / "summary.json").exists()
    else:
        r = cli.evaluate(ck, data, out)
        assert r["completed"] == 3 and r["clips"] == 40
    with pytest.raises(FileExistsError):
        cli.evaluate(ck, data, out)


@pytest.mark.parametrize("fault", ["asset", "csv", "policy", "perception"])
def test_preflight_rejects_unusable_inputs(tmp_path, fault):
    ck, data, out = fixture(tmp_path)
    if fault == "asset":
        (data / "g1_description/g1_29dof_rev_1_0.urdf").unlink()
    elif fault == "csv":
        (data / "lafan1_g1/clip0.csv").write_text("0,0,0")
    else:
        saved = torch.load(ck, weights_only=False)
        if fault == "policy":
            saved["policy"] = {"p": torch.ones(1)}
        else:
            saved["config"]["perception"] = True
        torch.save(saved, ck)
    with pytest.raises((ValueError, RuntimeError)):
        cli.evaluate(ck, data, out, dry_run=True)
    assert not out.exists()
