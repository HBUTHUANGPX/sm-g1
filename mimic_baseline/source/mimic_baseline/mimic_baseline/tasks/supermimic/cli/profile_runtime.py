"""Bounded performance audit. Outputs are benchmarks, never evaluation success."""

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import runpy
import sys
import time

from ..runtime.paths import training_entry


class BenchmarkComplete(Exception):
    pass


def portable_function(filename, line, name):
    value = filename.replace("\\", "/")
    if "/tasks/supermimic/" in value:
        value = value.split("/tasks/supermimic/", 1)[1]
    elif "/site-packages/" in value:
        value = value.split("/site-packages/", 1)[1]
    else:
        value = Path(value).name
    return f"{value}:{line}:{name}"



def run_training_window(trainer, original_train, kwargs, *, warmup_iterations,
                        timing, start, finish, synchronize, clock=time.perf_counter):
    """Observe original iteration boundaries; retain refill/log/save behavior."""
    original_collect = trainer.collect
    original_set_lr = trainer._set_lr
    previous = {name: trainer.__dict__.get(name) for name in ("collect", "_set_lr")}
    existed = {name: name in trainer.__dict__ for name in previous}
    first_iteration = trainer.iteration
    boundary = None
    measured_start = None
    timing["iteration_seconds"] = []

    def refill(count):
        begin = clock()
        result = original_collect(count)
        synchronize()
        timing["refill_seconds"] = clock() - begin
        trainer.collect = original_collect
        return result

    def set_lr(iteration):
        nonlocal boundary, measured_start
        if iteration >= first_iteration + warmup_iterations:
            if boundary is None:
                start()
                boundary = clock()
                measured_start = iteration
            else:
                synchronize()
                now = clock()
                timing["iteration_seconds"].append(now - boundary)
                boundary = now
        return original_set_lr(iteration)

    trainer.collect = refill
    trainer._set_lr = set_lr
    try:
        return original_train(trainer, **kwargs)
    finally:
        for name in previous:
            if existed[name]:
                setattr(trainer, name, previous[name])
            else:
                delattr(trainer, name)
        if boundary is not None:
            synchronize()
            timing["iteration_seconds"].append(clock() - boundary)
            timing["measured_training_iterations"] = trainer.iteration - measured_start
            finish()


def audit(
    checkpoint,
    data_root,
    output,
    *,
    target="training",
    iterations=5,
    steps=256,
    actuators="reference",
    kinematics="reference",
    profile="full_clean",
    verify_state_cache=False,
    warmup_iterations=0,
    cprofile=True,
    fk_chunk_size=4096,
):
    if (
        target not in ("training", "evaluation")
        or not 1 <= iterations <= 10
        or not 1 <= steps <= 2048
    ):
        raise ValueError(
            "Bounded audit requires 1..10 training iterations and 1..2048 evaluation steps"
        )
    if not 0 <= warmup_iterations < iterations:
        raise ValueError("warmup iterations must satisfy 0 <= warmup < iterations")
    if target == "evaluation" and warmup_iterations:
        raise ValueError("Evaluation requires zero warmup iterations")
    if actuators not in ("reference", "grouped") or kinematics not in ("reference", "batched", "bounded"):
        raise ValueError("Unsupported actuator or kinematics execution")
    if not isinstance(fk_chunk_size, int) or isinstance(fk_chunk_size, bool) or fk_chunk_size < 1:
        raise ValueError("fk_chunk_size must be a positive integer")
    if verify_state_cache and target != "evaluation":
        raise ValueError("State-cache validation requires evaluation")
    import torch
    from .evaluation_job import source_identity
    from ..runtime.paths import project_root

    source = source_identity(project_root())
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "supermimic-g1-v1":
        raise ValueError("Expected ManagerBased G1 checkpoint")
    start_iteration = saved["iteration"]
    config = saved["config"]
    del saved
    from ..robots.g1.kinematics import G1Kinematics

    original_fk = G1Kinematics.forward
    from ..envs.manager_based import SuperMimicRLEnv

    state_checks = 0
    from ..envs.evaluation import G1EvaluationEnv

    original_state = G1EvaluationEnv.state
    if verify_state_cache:
        def checked_state(env):
            nonlocal state_checks
            actual = original_state(env)
            if env._stepping and env._post_state is not None:
                fresh = SuperMimicRLEnv.state(env)
                for a, b in zip(actual, fresh, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                state_checks += 1
            return actual

    profiler = cProfile.Profile() if cprofile else None
    timing = {}
    old_argv = sys.argv

    def start():
        torch.cuda.synchronize()
        timing["pre_measurement_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        timing["pre_measurement_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        torch.cuda.reset_peak_memory_stats()
        timing["start"] = time.perf_counter()
        if profiler is not None:
            profiler.enable()

    def finish():
        if profiler is not None:
            profiler.disable()
        torch.cuda.synchronize()
        timing["measured_seconds"] = time.perf_counter() - timing.pop("start")
        timing["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        timing["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        for kind in ("allocated", "reserved"):
            timing[f"whole_audit_peak_{kind}_gib"] = max(
                timing[f"pre_measurement_peak_{kind}_gib"], timing[f"peak_{kind}_gib"]
            )

    command = [
        str(training_entry()),
        "--mode",
        "train" if target == "training" else "evaluate",
        "--resume",
        str(checkpoint),
        "--data-root",
        str(Path(data_root).resolve()),
        "--log-dir",
        str(output / "run"),
    ]
    if target == "training":
        from ..learning.trainer import Trainer

        original_train = Trainer.train

        def train(trainer, **kwargs):
            return run_training_window(
                trainer, original_train, kwargs, warmup_iterations=warmup_iterations,
                timing=timing, start=start, finish=finish,
                synchronize=torch.cuda.synchronize,
            )

        command += [
            "--iterations", str(start_iteration + iterations),
            "--training-execution", actuators,
            "--kinematics-execution", "reference" if kinematics == "batched" else kinematics,
            "--fk-chunk-size", str(fk_chunk_size),
        ]
    else:
        from .. import evaluation

        original_evaluate = evaluation.evaluate_tracking

        def evaluate(env, *args, **kwargs):
            original_step = env.step
            count = 0

            def step(action):
                nonlocal count
                if count >= steps:
                    raise BenchmarkComplete()
                count += 1
                return original_step(action)

            env.step = step
            start()
            try:
                return original_evaluate(env, *args, **kwargs)
            finally:
                finish()
                timing["measured_steps"] = count
                env.step = original_step

        command += [
            "--kinematics-execution",
            kinematics,
            "--fk-chunk-size", str(fk_chunk_size),
            "--evaluation-execution",
            actuators,
            "--evaluation-profile",
            profile,
        ]
    error = None
    try:
        # Install hooks only after all setup/imports have succeeded, inside the
        # restoration scope. Import failures must not leak patches to callers.
        sys.argv = command
        if target == "training" and kinematics == "batched":
            G1Kinematics.forward = G1Kinematics.forward_fast
        if target == "training":
            Trainer.train = train
        else:
            evaluation.evaluate_tracking = evaluate
        if verify_state_cache:
            G1EvaluationEnv.state = checked_state
        try:
            runpy.run_path(command[0], run_name="__main__")
        except BenchmarkComplete:
            pass
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        sys.argv = old_argv
        G1EvaluationEnv.state = original_state
        G1Kinematics.forward = original_fk
        if target == "training":
            Trainer.train = original_train
        else:
            evaluation.evaluate_tracking = original_evaluate
        rows = []
        if profiler is not None:
            profiler.create_stats()
            if profiler.stats:
                profiler.dump_stats(str(output / "cpu.prof"))
                stats = pstats.Stats(profiler)
                rows = [
                    {
                        "function": portable_function(*key),
                        "calls": value[1],
                        "self_seconds": value[2],
                        "cumulative_seconds": value[3],
                    }
                    for key, value in stats.stats.items()
                ]
                rows.sort(key=lambda r: r["cumulative_seconds"], reverse=True)
        with checkpoint.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        result = {
            "benchmark_only": True,
            "cprofile_enabled": cprofile,
            "warmup_iterations": warmup_iterations,
            "fk_chunk_size": fk_chunk_size,
            "experimental_training_batched_patch": target == "training" and kinematics == "batched",
            "state_cache_verification": {"enabled": verify_state_cache, "checked_reads": state_checks},
            "target": target,
            "source": source,
            "checkpoint_sha256": digest,
            "checkpoint_iteration": start_iteration,
            "actuators": actuators,
            "kinematics": kinematics,
            "profile": profile if target == "evaluation" else None,
            "training_iterations": iterations if target == "training" else None,
            "training_shape": {
                k: config.get(k)
                for k in (
                    "num_envs",
                    "wm_batch",
                    "wm_batches",
                    "policy_batch",
                    "horizon",
                    "buffer_steps",
                    "warmup_steps",
                    "log_every",
                )
            },
            "timing": timing,
            "error": error,
            "cpu_top": rows[:100],
        }
        (output / "profile.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "cpu_top"}), flush=True
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--target", choices=["training", "evaluation"], default="training")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--warmup-iterations", type=int, default=0)
    p.add_argument("--fk-chunk-size", type=int, default=4096)
    p.add_argument("--cprofile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--actuators", choices=["reference", "grouped"], default="reference")
    p.add_argument(
        "--kinematics", choices=["reference", "batched", "bounded"], default="reference"
    )
    p.add_argument("--verify-state-cache", action="store_true")
    p.add_argument("--profile", choices=["strict", "full_clean"], default="full_clean")
    audit(**vars(p.parse_args()))
