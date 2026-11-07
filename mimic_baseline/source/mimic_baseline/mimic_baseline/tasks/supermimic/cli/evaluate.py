"""Evaluate a ManagerBased G1 checkpoint on exactly all 40 LAFAN1 clips."""

from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys
from ..runtime.paths import training_entry
from ..evaluation.profiles import PROFILES, PROTOCOL_VERSION


def validate_results(payload, expected_names, *, profile="strict", expected_steps=None):
    if profile not in PROFILES:
        raise ValueError(f"Unknown evaluation profile: {profile}")
    rows = payload["clips"]
    if len(rows) != 40 or {r["name"] for r in rows} != set(expected_names):
        raise ValueError("Evaluation must contain exactly all 40 source clips")
    for row in rows:
        if any(
            isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()
        ):
            raise ValueError(f"Nonfinite evaluation metrics: {row['name']}")
    if profile == "full_clean":
        if expected_steps is None or set(expected_steps) != set(expected_names):
            raise ValueError(
                "Clean evaluation requires original source trajectory lengths"
            )
        for row in rows:
            expected = expected_steps[row["name"]]
            if (
                row.get("steps") != expected
                or row.get("full_clip_frames") != expected + 1
                or row.get("terminated") is not False
                or row.get("covered_to_end") is not True
                or "tracking_success" not in row
                or row["tracking_success"] is not None
            ):
                raise ValueError(
                    f"Invalid full-clean trajectory coverage: {row['name']}"
                )
        return {
            "clips": len(rows),
            "covered_clips": len(rows),
            "coverage_rate": 1.0,
            "tracking_success": None,
            "full_trajectory_is_tracking_success": False,
            "mean_normalized_return": sum(r["normalized_return"] for r in rows)
            / len(rows),
        }
    return {
        "clips": len(rows),
        "completed": sum(bool(r["completed"]) for r in rows),
        "completion_rate": sum(bool(r["completed"]) for r in rows) / len(rows),
        "mean_normalized_return": sum(r["normalized_return"] for r in rows) / len(rows),
        "failed_clips": [r["name"] for r in rows if not r["completed"]],
    }


def preflight_inputs(saved, data_root):
    """CPU validation of source data, required assets and strict policy ABI."""
    import numpy as np
    import torch
    from ..learning.config import TrainingConfig
    from ..learning.models import Policy
    from ..runtime.configuration import restore_run_config, resolve_backend

    values, small = restore_run_config(saved, {"device": "cpu"})
    cfg = TrainingConfig(**values)
    cfg.validate()
    resolve_backend(None, saved)
    with torch.random.fork_rng(devices=[]):
        kwargs = {"hidden": (64, 64)} if small else {}
        policy = Policy(
            93,
            73 * cfg.goal_frames,
            hist=cfg.hist,
            action_dim=29,
            actor_relative_orientation=cfg.actor_relative_orientation,
            **kwargs,
        )
        policy.load_state_dict(saved["policy"], strict=True)
    for relative in (
        "g1_description/g1_29dof_rev_1_0.urdf",
        "g1_usd/g1_29dof/g1_29dof.usda",
        "ground/default_ground_plane.usda",
    ):
        asset = data_root / relative
        if not asset.is_file() or asset.stat().st_size == 0:
            raise ValueError(f"Missing or empty required asset: {relative}")
    expected_steps = {}
    for source in sorted((data_root / "lafan1_g1").glob("*.csv")):
        raw = np.loadtxt(source, delimiter=",", dtype=np.float64)
        if (
            raw.ndim != 2
            or raw.shape[1] != 36
            or raw.shape[0] < 2
            or not np.isfinite(raw).all()
        ):
            raise ValueError(f"Expected finite [T>=2,36] G1 LAFAN1 CSV: {source.name}")
        if (np.linalg.norm(raw[:, 3:7], axis=1) < 1e-8).any():
            raise ValueError(f"Zero root quaternion: {source.name}")
        # Same original 30 Hz -> 50 Hz endpoint convention as convert_clip.
        expected_steps[source.stem] = int(np.floor((len(raw) - 1) / 30 / 0.02 + 1e-9))
    return expected_steps


def evaluate(checkpoint, data_root, output, *, dry_run=False, profile="strict"):
    import torch

    checkpoint, data_root, output = (
        Path(p).resolve() for p in (checkpoint, data_root, output)
    )
    if profile not in (*PROFILES, "both"):
        raise ValueError(f"Unknown evaluation profile: {profile}")
    if output.exists():
        raise FileExistsError(f"Refusing to reuse evaluation output: {output}")
    if profile == "both":
        results = {}
        for selected in PROFILES:
            results[selected] = evaluate(
                checkpoint,
                data_root,
                output / selected,
                dry_run=dry_run,
                profile=selected,
            )
        return {
            "profile": "both",
            "protocol_version": PROTOCOL_VERSION,
            "profiles": results,
        }
    names = sorted(p.stem for p in (data_root / "lafan1_g1").glob("*.csv"))
    if len(names) != 40:
        raise ValueError("Expected exactly 40 LAFAN1 source clips")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        saved.get("format") != "supermimic-g1-v1"
        or saved["config"].get("robot_name") != "g1"
    ):
        raise ValueError("Expected an original ManagerBased G1 checkpoint")
    if not saved.get("policy") or not all(
        torch.isfinite(v).all() for v in saved["policy"].values()
    ):
        raise ValueError("Missing or nonfinite policy parameters")
    expected_steps = preflight_inputs(saved, data_root)
    metadata = {
        "checkpoint": checkpoint.name,
        "checkpoint_iteration": saved["iteration"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "backend": saved.get("provenance", {}).get("backend", "ovphysx"),
        "expected_clips": names,
        "profile": profile,
        "protocol_version": PROTOCOL_VERSION,
        "protocol": (
            "native tracking failure termination; original unpadded clip lengths"
            if profile == "strict"
            else "full original trajectories; termination and random actuator delay disabled; coverage is not tracking success"
        ),
        "expected_steps": expected_steps,
        "recovery_enabled": saved["config"].get("recovery_enabled", False),
        "action_delay": {
            "actuator_type": saved["config"].get("actuator_type"),
            "max_delay": 0
            if profile == "full_clean"
            else saved["config"].get("max_delay"),
        },
        "seed": saved["config"].get("seed", 0),
    }
    del saved
    command = [
        sys.executable,
        "-u",
        str(training_entry()),
        "--mode",
        "evaluate",
        "--resume",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--log-dir",
        str(output),
        "--evaluation-profile",
        profile,
    ]
    if dry_run:
        return {"preflight_passed": True, "command": command, **metadata}
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    try:
        with (output / "evaluation.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = validate_results(
            json.loads((output / "perclip.json").read_text(encoding="utf-8")),
            names,
            profile=profile,
            expected_steps=expected_steps,
        )
        summary.update(metadata)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps({"error": str(error)}, indent=2), encoding="utf-8"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(*PROFILES, "both"),
        default="strict",
        help="Strict tracking or full clean trajectory coverage; both runs sequentially",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint and data without launching the simulator",
    )
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
