"""Run a reproducible evaluation job and generate a validated report."""

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..runtime.paths import project_root


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(root):
    root = Path(root)
    manifest_path = root / "release-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
        if not files or not manifest.get("source_revision"):
            raise ValueError("Incomplete release identity")
        for name, digest in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
                raise ValueError(f"Release file mismatch: {name}")
        # Every executed project Python file must belong to the verified release.
        for directory in (root / "scripts", root / "mimic_baseline"):
            for path in directory.rglob("*.py"):
                if path.relative_to(root).as_posix() not in files:
                    raise ValueError("Unmanifested Python source in release")
        return {
            "kind": "verified_release",
            "source_revision": manifest["source_revision"],
            "manifest_sha256": sha256(manifest_path),
        }

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(
            "Commit source changes before evaluation; dirty provenance is not reproducible"
        )
    return {"kind": "clean_git", "source_revision": git("rev-parse", "HEAD")}


def select_checkpoint(explicit, checkpoint_root):
    if explicit:
        checkpoint = Path(explicit).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    if not checkpoint_root:
        raise ValueError("Provide --checkpoint or checkpoint_root in local config")
    root = Path(checkpoint_root).expanduser().resolve()
    candidates = []
    for directory in (root, root / "train"):
        for path in directory.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            if match and path.is_file():
                candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError(
            "No model_N.pt checkpoint in checkpoint_root or its train directory"
        )
    latest = max(n for n, _ in candidates)
    matches = [p for n, p in candidates if n == latest]
    if len(matches) != 1:
        raise ValueError("Ambiguous latest checkpoint; specify --checkpoint")
    return matches[0]


def run_job(
    config,
    *,
    checkpoint=None,
    profile="both",
    execution="grouped",
    scheduling="sequential",
    kinematics_execution="reference",
):
    from ..evaluation.report import write_report
    from .evaluate import evaluate

    config_path = Path(config).expanduser().resolve()
    settings = json.loads(config_path.read_text())
    unknown = set(settings) - {"data_root", "checkpoint_root", "output_root"}
    if unknown:
        raise ValueError(
            f"Unknown config keys: {sorted(unknown)}; use the active Python interpreter"
        )

    def resolve(value):
        path = Path(value).expanduser()
        return (
            (config_path.parent / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )

    selected = select_checkpoint(
        checkpoint,
        resolve(settings["checkpoint_root"])
        if settings.get("checkpoint_root")
        else None,
    )
    source = source_identity(project_root())
    identity = {
        "checkpoint": selected.name,
        "checkpoint_sha256": sha256(selected),
        "source": source,
        "execution": execution,
        "kinematics_execution": kinematics_execution,
        "scheduling": scheduling,
        "profile": profile,
    }
    output = resolve(settings["output_root"]) / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + selected.stem
        + "-"
        + uuid4().hex[:10]
    )
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def status(state, **extra):
        value = {
            **identity,
            "status": state,
            "elapsed_seconds": time.monotonic() - start,
            **extra,
        }
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "status.json")

    status("running")
    print(f"Evaluation output: {output}", flush=True)
    try:
        result = evaluate(
            selected,
            resolve(settings["data_root"]),
            output / "evaluation"
            if profile == "both"
            else output / "evaluation" / profile,
            profile=profile,
            execution=execution,
            scheduling=scheduling,
            kinematics_execution=kinematics_execution,
        )
        if (
            sha256(selected) != identity["checkpoint_sha256"]
            or source_identity(project_root()) != source
        ):
            raise ValueError("Checkpoint or source changed during evaluation")
        results = result["profiles"] if profile == "both" else {profile: result}
        identity["evaluation_wall_seconds"] = time.monotonic() - start
        identity["profile_elapsed_seconds"] = {
            name: summary.get("elapsed_seconds") for name, summary in results.items()
        }
        write_report(output, results, identity)
        status("complete")
        print((output / "report.md").read_text(), flush=True)
    except BaseException as error:
        status("failed", error=str(error))
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics-execution", choices=("reference", "batched"), default="reference")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--profile", choices=("both", "strict", "full_clean"), default="both"
    )
    parser.add_argument(
        "--execution", choices=("grouped", "reference"), default="grouped"
    )
    parser.add_argument(
        "--scheduling", choices=("sequential", "parallel"), default="sequential"
    )
    run_job(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
