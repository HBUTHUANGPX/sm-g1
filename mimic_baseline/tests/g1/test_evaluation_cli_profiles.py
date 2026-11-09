"""Profile routing and full-trajectory evidence must not imply tracking success."""

import json
from pathlib import Path
import pytest
from mimic_baseline.tasks.supermimic.cli import evaluate as cli
from test_evaluation_cli import fixture


def clean_rows():
    return [
        dict(
            name=f"clip{i}",
            steps=1,
            full_clip_frames=2,
            terminated=False,
            covered_to_end=True,
            tracking_success=None,
            completed=True,
            normalized_return=0.5,
        )
        for i in range(40)
    ]


@pytest.mark.parametrize(
    "fault",
    ["steps", "terminated", "covered_to_end", "full_clip_frames", "tracking_success"],
)
def test_clean_rejects_incomplete_or_ambiguous_evidence(fault):
    rows = clean_rows()
    rows[0][fault] = {
        "steps": 2,
        "terminated": True,
        "covered_to_end": False,
        "full_clip_frames": 3,
        "tracking_success": True,
    }[fault]
    with pytest.raises(ValueError):
        cli.validate_results(
            {"clips": rows},
            [r["name"] for r in rows],
            profile="full_clean",
            expected_steps={r["name"]: 1 for r in rows},
        )


def test_clean_reports_coverage_without_tracking_completion():
    rows = clean_rows()
    result = cli.validate_results(
        {"clips": rows},
        [r["name"] for r in rows],
        profile="full_clean",
        expected_steps={r["name"]: 1 for r in rows},
    )
    assert result["covered_clips"] == 40 and result["coverage_rate"] == 1
    assert result["tracking_success"] is None
    assert "completed" not in result and "completion_rate" not in result


def test_both_dry_run_preserves_filesystem_and_records_effective_delay(tmp_path):
    ck, data, out = fixture(tmp_path)
    result = cli.evaluate(ck, data, out, profile="both", dry_run=True)
    assert not out.exists()
    for profile, entry in result["profiles"].items():
        assert entry["protocol_version"] == 2 and entry["profile"] == profile
        assert entry["command"][-2:] == ["--evaluation-profile", profile]
        assert str(out / profile) in entry["command"]
    assert result["profiles"]["full_clean"]["action_delay"]["max_delay"] == 0


def test_both_runs_sequentially_and_preserves_strict_summary(tmp_path, monkeypatch):
    ck, data, out = fixture(tmp_path)
    calls = []

    def run(command, **kwargs):
        profile = command[-1]
        calls.append(profile)
        dest = Path(command[command.index("--log-dir") + 1])
        rows = clean_rows()
        (dest / "perclip.json").write_text(json.dumps({"clips": rows}))

    monkeypatch.setattr(cli.subprocess, "run", run)
    result = cli.evaluate(ck, data, out, profile="both")
    assert calls == ["strict", "full_clean"]
    assert result["profiles"]["strict"]["completed"] == 40
    assert result["profiles"]["full_clean"]["covered_clips"] == 40
    with pytest.raises(FileExistsError):
        cli.evaluate(ck, data, out, profile="both")


def test_both_stops_after_strict_failure(tmp_path, monkeypatch):
    ck, data, out = fixture(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command[-1])
        raise RuntimeError("failed strict")

    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        cli.evaluate(ck, data, out, profile="both")
    assert calls == ["strict"] and not (out / "full_clean").exists()


def test_clean_uses_resampled_source_length_not_claimed_output_length(
    tmp_path, monkeypatch
):
    import numpy as np

    ck, data, out = fixture(tmp_path)
    raw = np.zeros((4, 36))
    raw[:, 6] = 1
    np.savetxt(data / "lafan1_g1/clip0.csv", raw, delimiter=",")

    def run(command, **kwargs):
        # A self-consistent output claiming only two frames is still incomplete.
        (out / "perclip.json").write_text(json.dumps({"clips": clean_rows()}))

    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(ValueError, match="coverage"):
        cli.evaluate(ck, data, out, profile="full_clean")
    assert (out / "failure.json").exists() and not (out / "summary.json").exists()
    manifest = json.loads((out / "evaluation_manifest.json").read_text())
    assert manifest["expected_steps"]["clip0"] == 5


def test_grouped_execution_is_explicit_in_child_command(tmp_path):
    ck, data, out = fixture(tmp_path)
    result = cli.evaluate(
        ck, data, out, profile="both", execution="grouped", dry_run=True
    )
    for entry in result["profiles"].values():
        command = entry["command"]
        assert command[command.index("--evaluation-execution") + 1] == "grouped"
        assert entry["execution"] == "grouped"


def test_grouped_rejects_unsupported_actuator_before_starting_job(tmp_path):
    import torch
    ck,data,out=fixture(tmp_path)
    saved=torch.load(ck,weights_only=False);saved['config']['actuator_type']='implicit';torch.save(saved,ck)
    with pytest.raises(ValueError,match='delayed_pd'):
        cli.evaluate(ck,data,out,execution='grouped',dry_run=True)
    assert not out.exists()


def test_batched_kinematics_is_forwarded_to_both_profiles(tmp_path):
    ck, data, out = fixture(tmp_path)
    result = cli.evaluate(ck, data, out, profile='both', dry_run=True,
                          kinematics_execution='batched')
    for entry in result['profiles'].values():
        assert entry['kinematics_execution'] == 'batched'
        command = entry['command']
        assert command[command.index('--kinematics-execution') + 1] == 'batched'


def test_batched_kinematics_rejected_for_training_before_simulator(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(root / 'scripts/train_g1.py'), '--mode', 'train',
         '--data-root', str(tmp_path), '--log-dir', str(tmp_path / 'run'),
         '--kinematics-execution', 'batched'],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert 'supported only in evaluate mode' in result.stderr
    assert not (tmp_path / 'run').exists()


def test_training_execution_flags_are_mode_checked():
    import subprocess
    import sys
    from pathlib import Path
    entry = Path(__file__).resolve().parents[3] / "scripts/train_g1.py"
    common = [sys.executable, str(entry), "--data-root", "missing", "--log-dir", "unused"]
    for extra, message in [
        (["--mode", "prepare", "--training-execution", "grouped"], "training-execution"),
        (["--mode", "evaluate", "--kinematics-execution", "bounded"], "bounded"),
        (["--mode", "train", "--kinematics-execution", "bounded", "--fk-chunk-size", "0"], "positive"),
    ]:
        run = subprocess.run(common + extra, capture_output=True, text=True)
        assert run.returncode != 0 and message in run.stderr
        assert "Traceback" not in run.stderr


def test_bounded_training_flags_are_documented():
    import subprocess
    import sys
    from pathlib import Path
    entry = Path(__file__).resolve().parents[3] / "scripts/train_g1.py"
    help_text = subprocess.check_output([sys.executable, str(entry), "--help"], text=True)
    assert "--training-execution" in help_text
    assert "--fk-chunk-size" in help_text
    assert "bounded" in help_text
