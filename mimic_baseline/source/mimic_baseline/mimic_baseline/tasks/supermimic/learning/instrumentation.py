"""Original timing and JSONL instrumentation."""

from __future__ import annotations
import contextlib


import json


import os


import time


from pathlib import Path


from typing import Any, Iterator


import torch


class Profiler:
    """CUDA-event profiler with one synchronization per log interval."""

    def __init__(self, device: torch.device):
        self.device = device
        self._events: dict[str, list[tuple[Any, Any]]] = {}

    @contextlib.contextmanager
    def record(self, name: str) -> Iterator[None]:
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._events.setdefault(name, []).append((start, end))
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                self._events.setdefault(name, []).append((start, time.perf_counter()))

    def drain(self) -> dict[str, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result: dict[str, float] = {}
        for name, events in self._events.items():
            if self.device.type == "cuda":
                values = [start.elapsed_time(end) for start, end in events]
            else:
                values = [(end - start) * 1e3 for start, end in events]
            result[f"Performance/{name}"] = sum(values) / max(len(values), 1)
        self._events.clear()
        return result


class JsonlLogger:
    """Small local logger; SwanLab can be attached without changing Trainer."""

    def __init__(self, path: str | os.PathLike[str], swanlab: Any | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.swanlab = swanlab

    def log(self, metrics: dict[str, float], step: int) -> None:
        row = {"iteration": step, **metrics}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self.swanlab is not None:
            self.swanlab.log(metrics, step=step)

    def close(self) -> None:
        if self.swanlab is not None and hasattr(self.swanlab, "finish"):
            self.swanlab.finish()
            self.swanlab = None
