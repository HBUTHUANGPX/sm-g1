"""Evaluate saved Newton checkpoints without keeping an idle simulator on the GPU."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--interval", type=int, default=1000)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    exit_file = run.with_suffix(".exit")
    evaluator = Path(__file__).resolve().parents[1] / "supermimic" / "evaluate.py"
    while True:
        finished = exit_file.exists()
        checkpoints = sorted(run.glob("ckpt_*.pt"))
        for ckpt in checkpoints:
            iteration = int(ckpt.stem.split("_")[-1])
            if iteration % args.interval and not (finished and ckpt == checkpoints[-1]):
                continue
            result_path = ckpt.with_suffix(".eval.json")
            error_path = ckpt.with_suffix(".eval-error.json")
            if result_path.exists() or (error_path.exists() and not finished):
                continue
            log_path = ckpt.with_suffix(".eval.log")
            with log_path.open("wb") as log:
                result = subprocess.run(
                    [sys.executable, str(evaluator), "--backend", "newton", "--viz", "none",
                     "--run_dir", str(run), "--ckpt", str(ckpt)],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            output = log_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"(?m)^\{\r?\n\s*\"norm_return\":", output)
            if result.returncode or not match:
                save_json(error_path, {"iteration": iteration, "returncode": result.returncode, "log": log_path.name})
                print(f"Evaluation failed at {iteration}; see {log_path.name}", flush=True)
                continue
            metrics, _ = json.JSONDecoder().raw_decode(output[match.start():])
            metrics.update(iteration=iteration, checkpoint=ckpt.name)
            save_json(result_path, metrics)
            reports = [json.loads(p.read_text(encoding="utf-8")) for p in run.glob("ckpt_*.eval.json")]
            best = max(reports, key=lambda m: (-m["n_failed"], m["norm_return"]))
            save_json(run / "best_eval.json", best)
            print(f"Iteration {iteration}: normalized return {metrics['norm_return']:.4f}; "
                  f"{40 - metrics['n_failed']}/40 clips completed.", flush=True)
        if finished:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
