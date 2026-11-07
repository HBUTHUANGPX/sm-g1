"""Bounded-frequency progress, with no per-step disk writes."""

import json
import time
from pathlib import Path


class RolloutProgress:
    def __init__(self, output):
        self.output = Path(output)
        self.start = time.perf_counter()

    def __call__(self, step, total, active, steps):
        elapsed = time.perf_counter() - self.start
        remaining = int(active.sum().item())
        payload = {
            "phase": "rollout_complete" if remaining == 0 else "rolling_out",
            "step": step,
            "max_steps": total,
            "active_clips": remaining,
            "evaluated_transitions": int(steps.sum().item()),
            "elapsed_seconds": elapsed,
            "vector_steps_per_second": step / elapsed if elapsed else 0,
            "eta_upper_seconds": (total - step) * elapsed / step if step else None,
        }
        temp = self.output / "progress.tmp"
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.output / "progress.json")
        print(json.dumps({"evaluation_progress": payload}), flush=True)
