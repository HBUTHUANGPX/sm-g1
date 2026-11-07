"""Verify a finished SuperMimic training run and its resumable checkpoint."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch


def verify(run: Path, backend: str, iterations: int):
    args = json.loads((run / "args.json").read_text())
    if (args["backend"], args["iterations"]) != (backend, iterations):
        raise ValueError("Run arguments do not match the requested backend and iteration count")
    with (run / "train_log.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    numbers = [int(row["iteration"]) for row in rows]
    if not numbers or numbers[-1] != iterations or numbers != sorted(set(numbers)):
        raise ValueError("Training log does not end at the expected iteration or has duplicate rows")
    if any(not math.isfinite(float(v)) for row in rows for v in row.values()):
        raise ValueError("Training log contains non-finite metrics")
    step_count = int(rows[-1]["sim_steps"])
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    if (ck["iteration"], ck["sim_steps"], ck["args"]["backend"]) != (iterations, step_count, backend):
        raise ValueError("Final checkpoint does not agree with the training log")
    for key in ("wm", "pi", "critic"):
        if not ck[key] or any(not torch.isfinite(t).all().item() for t in ck[key].values()):
            raise ValueError(f"Final checkpoint has empty or non-finite {key} weights")
    if not all(ck[key]["state"] for key in ("opt_wm", "opt_pi", "opt_v")):
        raise ValueError("Final checkpoint is missing optimizer updates")
    policy = torch.load(run / f"ckpt_{iterations:06d}.pt", map_location="cpu", weights_only=True)
    if policy["iteration"] != iterations or any(
        not torch.equal(t, policy[key][name]) for key in ("wm", "pi") for name, t in ck[key].items()
    ):
        raise ValueError("Numbered checkpoint does not match the final resumable checkpoint")
    return {
        "backend": backend, "iterations": iterations, "sim_steps": step_count,
        "wall_minutes": round(ck["wall_time"] / 60, 2), "last_logged_reward": float(rows[-1]["sim_rew"]),
        "checkpoint": str(run / f"ckpt_{iterations:06d}.pt"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.backend, args.iterations), indent=2))
