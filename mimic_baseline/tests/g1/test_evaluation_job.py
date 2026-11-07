"""Evaluation orchestration and reports need no simulator."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mimic_baseline.tasks.supermimic.cli import evaluation_job as job
from mimic_baseline.tasks.supermimic.evaluation.report import build_report

ROOT = Path(__file__).resolve().parents[3]


def fixtures():
    payloads = {
        p: json.loads(
            (
                ROOT / f"docs/results/2026-11-08-g1-6600-{p.replace('_', '-')}.json"
            ).read_text()
        )
        for p in ("strict", "full_clean")
    }
    results = {p: x["summary"] for p, x in payloads.items()}
    identity = {
        "checkpoint_sha256": results["strict"]["checkpoint_sha256"],
        "source": {"source_revision": "fixture"},
        "execution": "reference",
    }
    return results, payloads, identity


def test_report_real_results():
    results, payloads, identity = fixtures()
    summary, report = build_report(results, payloads, identity)
    assert summary["profiles"]["strict"]["completed"] == 6
    assert summary["profiles"]["full_clean"]["covered_clips"] == 40
    assert summary["full_trajectory_is_tracking_success"] is False
    for name in results["strict"]["expected_clips"]:
        assert report.count(f"| {name} |") == 2
    assert "tracking success not measured" in report


@pytest.mark.parametrize(
    "fault", ["hash", "missing_runtime", "after_delay", "coverage"]
)
def test_report_rejects_invalid_evidence(fault):
    results, payloads, identity = copy.deepcopy(fixtures())
    if fault == "hash":
        identity["checkpoint_sha256"] = "wrong"
    elif fault == "missing_runtime":
        payloads["full_clean"].pop("runtime_checks")
    elif fault == "after_delay":
        payloads["full_clean"]["runtime_checks"]["after_rollout"]["action_delay_ticks"][
            "bad"
        ] = [1]
    else:
        payloads["full_clean"]["clips"][0]["steps"] -= 1
    with pytest.raises(ValueError):
        build_report(results, payloads, identity)


def test_numeric_latest_and_ambiguity(tmp_path):
    (tmp_path / "train").mkdir()
    for name in ("model_9.pt", "model_100.pt", "model_final.pt"):
        (tmp_path / "train" / name).touch()
    assert job.select_checkpoint(None, tmp_path).name == "model_100.pt"
    (tmp_path / "model_100.pt").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        job.select_checkpoint(None, tmp_path)


def test_release_source_hash_checked(tmp_path):
    source = tmp_path / "scripts" / "main.py"
    source.parent.mkdir()
    source.write_text("pass\n")
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(
            {"source_revision": "abc", "files": {"scripts/main.py": job.sha256(source)}}
        )
    )
    assert job.source_identity(tmp_path)["kind"] == "verified_release"
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="mismatch"):
        job.source_identity(tmp_path)


def test_job_status_and_unique_outputs(tmp_path, monkeypatch):
    from mimic_baseline.tasks.supermimic.cli import evaluate as evaluator

    checkpoint = tmp_path / "model_6600.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "local.json"
    config.write_text(
        json.dumps(
            {"data_root": "data", "checkpoint_root": ".", "output_root": "outputs"}
        )
    )
    monkeypatch.setattr(job, "source_identity", lambda _: {"source_revision": "test"})

    def fake_evaluate(checkpoint, data_root, output, **kwargs):
        results, payloads, _ = fixtures()
        for profile, payload in payloads.items():
            results[profile]["checkpoint_sha256"] = job.sha256(checkpoint)
            directory = output / profile
            directory.mkdir(parents=True)
            runtime = payload.pop("runtime_checks")
            runtime["execution"] = kwargs["execution"]
            runtime["after_rollout"]["execution"] = kwargs["execution"]
            payload["metadata"] = {"evaluation": runtime}
            (directory / "runtime_profile.json").write_text(json.dumps(runtime))
            (directory / "perclip.json").write_text(json.dumps(payload))
        return {"profiles": results}

    monkeypatch.setattr(evaluator, "evaluate", fake_evaluate)
    first = job.run_job(config)
    second = job.run_job(config)
    assert first != second
    assert json.loads((first / "status.json").read_text())["status"] == "complete"

    def fail(*args, **kwargs):
        raise RuntimeError("simulation failed")

    monkeypatch.setattr(evaluator, "evaluate", fail)
    with pytest.raises(RuntimeError, match="simulation failed"):
        job.run_job(config)
    statuses = [
        json.loads(p.read_text())["status"]
        for p in (tmp_path / "outputs").glob("*/status.json")
    ]
    assert sorted(statuses) == ["complete", "complete", "failed"]


def test_help_without_simulator():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_g1_evaluation.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--execution" in result.stdout
