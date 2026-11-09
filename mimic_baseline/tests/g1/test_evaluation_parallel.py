"""Prove scheduling and failure draining without a simulator or GPU."""

import json
import threading
from pathlib import Path

import pytest
from mimic_baseline.tasks.supermimic.cli import evaluate as cli
from mimic_baseline.tasks.supermimic.evaluation.report import build_report
from test_evaluation_cli import fixture
from test_evaluation_cli_profiles import clean_rows
from test_evaluation_job import fixtures


@pytest.mark.parametrize("scheduling", ["sequential", "parallel"])
def test_two_complete_profiles_and_isolated_outputs(tmp_path, monkeypatch, scheduling):
    checkpoint, data, output = fixture(tmp_path)
    strict_started, clean_started, strict_finished = (
        threading.Event() for _ in range(3)
    )
    destinations = []

    def run(command, **kwargs):
        profile = command[-1]
        destination = Path(command[command.index("--log-dir") + 1])
        destinations.append(destination)
        if profile == "strict":
            strict_started.set()
            if scheduling == "parallel":
                assert clean_started.wait(10), "profiles did not overlap"
            else:
                assert not clean_started.is_set()
            strict_finished.set()
        else:
            assert strict_started.wait(10)
            if scheduling == "sequential":
                assert strict_finished.is_set()
            clean_started.set()
        (destination / "perclip.json").write_text(json.dumps({"clips": clean_rows()}))

    monkeypatch.setattr(cli.subprocess, "run", run)
    result = cli.evaluate(
        checkpoint, data, output, profile="both", scheduling=scheduling
    )
    assert result["scheduling"] == scheduling
    assert set(destinations) == {output / "strict", output / "full_clean"}
    assert result["profiles"]["strict"]["completed"] == 40
    assert result["profiles"]["full_clean"]["covered_clips"] == 40
    assert result["profiles"]["full_clean"]["tracking_success"] is None
    assert len({r["checkpoint_sha256"] for r in result["profiles"].values()}) == 1
    assert len({r["seed"] for r in result["profiles"].values()}) == 1


def test_parallel_failure_waits_for_running_sibling(tmp_path, monkeypatch):
    checkpoint, data, output = fixture(tmp_path)
    clean_started, strict_failed, clean_finished = (threading.Event() for _ in range(3))

    def run(command, **kwargs):
        destination = Path(command[command.index("--log-dir") + 1])
        if command[-1] == "strict":
            assert clean_started.wait(10)
            strict_failed.set()
            raise RuntimeError("strict simulation failed")
        clean_started.set()
        assert strict_failed.wait(10)
        (destination / "perclip.json").write_text(json.dumps({"clips": clean_rows()}))
        clean_finished.set()

    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="strict simulation failed"):
        cli.evaluate(checkpoint, data, output, profile="both", scheduling="parallel")
    assert clean_finished.is_set()
    assert (output / "strict/failure.json").is_file()
    assert not (output / "strict/summary.json").exists()
    assert (output / "full_clean/summary.json").is_file()


def test_both_report_rejects_missing_profile():
    results, payloads, identity = fixtures()
    identity.update(profile="both", scheduling="parallel")
    results.pop("strict")
    with pytest.raises(ValueError, match="requires strict and full_clean"):
        build_report(results, payloads, identity)
