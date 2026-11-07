"""Wait for a completed training run and its evaluation before evaluating both profiles."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .experiment import verify_source as verify_release, write_json


def _load_checkpoint(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def run_queue(
    wait_run,
    expected_training_revision,
    source_revision,
    target_iteration,
    data_root,
    output,
    *,
    timeout=48 * 3600,
    poll=60,
    sleep=time.sleep,
    clock=time.monotonic,
    verify_source=verify_release,
    load_checkpoint=_load_checkpoint,
    evaluate=None,
):
    """Read-only wait on the old run; own output and fresh release remain separate."""
    for revision in (expected_training_revision, source_revision):
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Expected full Git source revision")
    if target_iteration < 1 or timeout <= 0 or poll <= 0:
        raise ValueError("Iteration, timeout and poll must be positive")
    wait_run, data_root, output = (
        Path(p).resolve() for p in (wait_run, data_root, output)
    )
    if output == wait_run or output.is_relative_to(wait_run):
        raise ValueError("Queue output must be separate from the active run")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(
        phase="waiting",
        source_revision=source_revision,
        expected_training_revision=expected_training_revision,
        target_iteration=target_iteration,
        profile="both",
    )

    def record(phase, **values):
        state.update(
            phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **values
        )
        write_json(output / "status.json", state)

    started = clock()
    try:
        verify_source(source_revision)
        record("waiting")
        while True:
            if clock() - started >= timeout:
                raise TimeoutError(
                    "Timed out waiting for training and its existing evaluation"
                )
            status_path = wait_run / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("source_revision") != expected_training_revision:
                    raise ValueError("Training source revision mismatch")
                if status.get("target_iteration") != target_iteration:
                    raise ValueError("Training target iteration mismatch")
                phase = status.get("phase", "")
                if "failed" in phase:
                    raise ValueError(f"Upstream run failed: {phase}")
                if phase == "complete":
                    break
                if phase not in {"training", "evaluating"}:
                    raise ValueError(f"Unexpected upstream phase: {phase}")
            sleep(min(poll, max(0, timeout - (clock() - started))))
        if status.get("checkpoint_iteration") != target_iteration:
            raise ValueError("Completed checkpoint iteration mismatch")
        checkpoint = wait_run / "train" / f"model_{target_iteration}.pt"
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != status.get("checkpoint_sha256"):
            raise ValueError("Completed checkpoint digest mismatch")
        saved = load_checkpoint(checkpoint)
        if (
            saved.get("iteration") != target_iteration
            or saved.get("format") != "supermimic-g1-v1"
        ):
            raise ValueError("Checkpoint payload iteration/format mismatch")
        del saved
        verify_source(source_revision)
        record(
            "evaluating",
            checkpoint_sha256=digest,
            checkpoint_iteration=target_iteration,
        )
        if evaluate is None:
            from .evaluate import evaluate
        result = evaluate(checkpoint, data_root, output / "evaluation", profile="both")
        write_json(output / "result.json", result)
        record("complete")
        return state
    except BaseException as error:
        record("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-run", type=Path, required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-iteration", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timeout",
        type=float,
        default=48 * 3600,
        help="Maximum waiting time in seconds",
    )
    parser.add_argument(
        "--poll", type=float, default=60, help="Polling interval in seconds"
    )
    print(json.dumps(run_queue(**vars(parser.parse_args())), indent=2))
