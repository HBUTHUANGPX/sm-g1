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
