"""Original Trainer composition, initialization and iteration lifecycle."""

from __future__ import annotations
import contextlib


import json


import math


import os


import random


import time


from dataclasses import asdict, dataclass


from functools import partial


from pathlib import Path


from typing import Any, Callable, Iterator


import torch


import torch.nn as nn


from torch.utils.checkpoint import checkpoint


from mimic_baseline.tasks.supermimic.core.action_bounds import (
    ACTION_LIMIT as _COLLECTION_ACTION_LIMIT,
    clip_action,
)


from mimic_baseline.tasks.supermimic.learning.distributed import (
    DistributedContext,
    default_lr_scale,
)


from mimic_baseline.tasks.supermimic.core.mdp import (
    MAX_STEP_REWARD,
    RELATIVE_ORIENTATION_NOISE,
    relative_root_orientation,
)


from mimic_baseline.tasks.supermimic.data.motion import (
    GOAL_PER_FRAME,
    GOAL_PER_FRAME_LEGACY,
)


from mimic_baseline.tasks.supermimic.learning.models import WM_OUT


from mimic_baseline.tasks.supermimic.learning.perception import (
    CLEAN_DEPTH_CLIP_M,
    DEPTH_SHAPE,
    HEIGHT_SCAN_DIM,
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
    height_scan_orientation,
    height_scan_shape,
)


from mimic_baseline.tasks.supermimic.learning.perception_loss import (
    cap_decoder_physics_gradients,
)


from mimic_baseline.tasks.supermimic.core.recovery import (
    recovery_provenance,
    remaining_protection,
)


from mimic_baseline.tasks.supermimic.data.reference import ReferenceBatch


from mimic_baseline.tasks.supermimic.runtime.safety import PeriodicCudaCache


from .config import TrainingConfig
from .instrumentation import Profiler, JsonlLogger
from .common import _valid_depth, _as_modules
from .world_model_update import WorldModelUpdateMixin
from .policy_update import PolicyUpdateMixin
from .collection import CollectionMixin
from .checkpoint import CheckpointMixin

_PAPER = "SuperMimic, arXiv:2610.28480"
_SOURCE_COMMIT = "27126bddd74381d164118ef83ee78edb58484a6b"


class Trainer(
    WorldModelUpdateMixin, PolicyUpdateMixin, CollectionMixin, CheckpointMixin
):
    """Pure collect/fit-world-model/actor-critic loop over injected objects."""

    # 73-D layout: p(3), 6-D rotation(6), v(3), w(3), q(29), qd(29).
    WM_ERROR_FLOOR = (
        ((3e-3,) * 3 + (5e-3,) * 6 + (2e-2,) * 3 + (5e-2,) * 3)
        + (3e-3,) * 29
        + (5e-2,) * 29
    )

    # Latent floor, appended after the physical floor when perception is on.
    # ``tanh`` bounds the latent to [-1, 1], so a normalizer that has seen only
    # a narrow early distribution must not be allowed to divide the error by a
    # near-zero standard deviation and dominate the shared objective.
    LATENT_ERROR_FLOOR = 5e-2

    def __init__(
        self,
        env: Any,
        buffer: Any,
        motion: Any,
        mdp: Any,
        world_model: nn.Module,
        policy: nn.Module,
        critic: nn.Module,
        unflatten_state: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
        normalize_xi: Callable[[torch.Tensor], torch.Tensor],
        cfg: TrainingConfig,
        *,
        optimizers: tuple[
            torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer
        ]
        | None = None,
        perception: nn.Module | None = None,
        logger: Any | None = None,
        profiler: Profiler | None = None,
        provenance: dict[str, Any] | None = None,
        evaluation_callback: Callable[[Path], None] | None = None,
        distributed: DistributedContext | None = None,
        physics_health: Any | None = None,
    ):
        cfg.validate()
        self.cfg = cfg
        self.recurrent = cfg.perception and cfg.perception_arch == "recurrent"
        self.device = torch.device(cfg.device)
        self.physics_health = physics_health
        self.cuda_cache = PeriodicCudaCache(self.device, cfg.cuda_cache_clear_interval)
        self._perception_loss_metrics = {}
        if self.recurrent:
            ratio = cfg.perception_decoder_physics_ratio
            # Upload fixed run settings once; rollout logs reuse these GPU views.
            settings = torch.tensor(
                [
                    cfg.perception_height_weight,
                    cfg.perception_depth_weight,
                    ratio is not None,
                    0.0 if ratio is None else ratio,
                ],
                dtype=torch.float32,
                device=self.device,
            )
            self._perception_loss_metrics = dict(
                zip(
                    (
                        "Perception/height_weight",
                        "Perception/depth_weight",
                        "Perception/decoder_physics_cap_enabled",
                        "Perception/decoder_physics_ratio",
                    ),
                    settings.unbind(),
                    strict=True,
                )
            )
        self.world_size = distributed.world_size if distributed is not None else 1
        self.env = env
        self.buffer = buffer
        self.motion = mdp.motion
        self.reference = mdp.reference
        self.pmg = None
        self.sampler = getattr(env, "sampler", None)
        self.mdp = mdp
        for name, owner in (("env", env), ("buffer", buffer), ("mdp", mdp)):
            if bool(getattr(owner, "recovery_enabled", False)) != cfg.recovery_enabled:
                raise ValueError(
                    f"{name}.recovery_enabled must match config.recovery_enabled"
                )
        self.wm = world_model
        self.policy = policy
        if (
            bool(getattr(policy, "relative_orientation_dim", 0))
            != cfg.actor_relative_orientation
        ):
            raise ValueError(
                "policy relative_orientation input must match config.actor_relative_orientation"
            )
        self.critic = critic
        self.unflatten = unflatten_state
        self.normalize_xi = normalize_xi
        if optimizers is None:
            effective_lr = cfg.lr * cfg.lr_scale
            optimizers = (
                torch.optim.Adam(
                    self.wm.parameters(), lr=effective_lr, betas=(0.9, 0.999)
                ),
                torch.optim.Adam(
                    self.policy.parameters(), lr=effective_lr, betas=(0.9, 0.95)
                ),
                torch.optim.Adam(
                    self.critic.parameters(), lr=effective_lr, betas=(0.9, 0.999)
                ),
            )
        self.opt_wm, self.opt_policy, self.opt_critic = optimizers
        if cfg.perception:
            if perception is None:
                raise ValueError("config.perception requires a perception module")
            model_latent_dim = (
                world_model.driver_dim if self.recurrent else world_model.latent_dim
            )
            if model_latent_dim != cfg.perception_latent_dim:
                raise ValueError(
                    f"world model perception width {model_latent_dim} does not match "
                    f"config.perception_latent_dim {cfg.perception_latent_dim}"
                )
            if getattr(policy, "perception_dim", 0) != cfg.perception_latent_dim:
                raise ValueError(
                    f"policy perception_dim {getattr(policy, 'perception_dim', 0)} "
                    f"does not match config.perception_latent_dim "
                    f"{cfg.perception_latent_dim}"
                )
            if not getattr(buffer, "perception", False):
                raise ValueError(
                    "perception training requires a replay buffer allocated with "
                    "perception=True"
                )
            if getattr(perception, "latent_dim", None) != cfg.perception_latent_dim:
                raise ValueError(
                    f"perception latent_dim {getattr(perception, 'latent_dim', None)} "
                    f"does not match config.perception_latent_dim "
                    f"{cfg.perception_latent_dim}"
                )
            # The encoder/decoder stay in the world-model optimizer: they are
            # trained end to end with it, so one gradient clip and one learning
            # rate schedule cover the whole perception-conditioned predictor.
            self.opt_wm.add_param_group({"params": list(perception.parameters())})
        elif perception is not None:
            raise ValueError(
                "a perception module was supplied but config.perception is False"
            )
        self.perception = perception
        if self.recurrent:
            world_frame = cfg.height_scan_frame == "world"
            if (
                getattr(perception, "world_frame", False) != world_frame
                or getattr(critic, "world_frame", False) != world_frame
            ):
                raise ValueError(
                    "WM and critic height encoders must match config.height_scan_frame"
                )
            for name, owner in (
                ("perception", perception),
                ("critic", critic),
                ("buffer", buffer),
            ):
                if getattr(owner, "scan_dim", HEIGHT_SCAN_DIM) != cfg.height_scan_dim:
                    raise ValueError(
                        f"{name}.scan_dim must match config.height_scan_dim"
                    )
            env_cfg = getattr(env, "cfg", None)
            if env_cfg is not None and (
                tuple(getattr(env_cfg, "height_scan_size", HEIGHT_SCAN_SIZE))
                != tuple(cfg.height_scan_size)
                or getattr(env_cfg, "height_scan_resolution", HEIGHT_SCAN_RESOLUTION)
                != cfg.height_scan_resolution
            ):
                raise ValueError(
                    "environment height_scan geometry must match training config"
                )
        self.camera = None
        self.camera_geometry = None
        if self.recurrent:
            from mimic_baseline.tasks.supermimic.camera import CameraTransport

            self.camera = getattr(env, "camera_transport", None) or CameraTransport()
            if self.camera.stereo_mode != "off":
                self.camera_geometry = env.camera_geometry
        self.logger = logger
        self.profiler = profiler or Profiler(self.device)
        self.evaluation_callback = evaluation_callback
        self.distributed = distributed
        self.provenance = provenance or {
            "paper": _PAPER,
            "reference_commit": _SOURCE_COMMIT,
            "implementation": "independent RX-P2 package",
        }
        self.provenance["recovery"] = recovery_provenance(
            cfg.recovery_enabled,
            cfg.term_height,
            cfg.term_tilt,
            action_dim=getattr(self.motion, "action_dim", 27),
        )
        self.iteration = 0
        self.sim_steps = 0
        self.transitions = 0
        self.windows = 0
        self.global_sim_steps = 0
        self.global_transitions = 0
        self.global_windows = 0
        self.optimizer_steps = 0
        self.step_accumulator = 0.0
        self._current_lr = cfg.lr * cfg.lr_scale
        self._obs_history: torch.Tensor | None = None
        self._pending: dict[str, list[torch.Tensor]] = {}
        self._pending_counts: dict[str, list[torch.Tensor]] = {}
        self._start_time = time.monotonic()
        self._last_log_time = self._start_time
        self._last_log_iteration = 0
        self._floor = torch.tensor(
            self.WM_ERROR_FLOOR, dtype=torch.float32, device=self.device
        )
        if cfg.perception and not self.recurrent:
            self._floor = torch.cat(
                (
                    self._floor,
                    torch.full(
                        (cfg.perception_latent_dim,),
                        self.LATENT_ERROR_FLOOR,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            )
        action_dim = getattr(self.motion, "action_dim", 27)
        per_frame = 2 * action_dim + (6 if cfg.goal_mode == "state_relative" else 15)
        goal_dim = per_frame * cfg.goal_frames
        if (
            getattr(self.policy, "goal_norm", None) is not None
            and self.policy.goal_norm.mean.numel() != goal_dim
        ):
            raise ValueError(
                f"policy goal dimension {self.policy.goal_norm.mean.numel()} does not match {per_frame}*goal_frames={goal_dim}"
            )

    def _goal(
        self, reference: ReferenceBatch, state: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return reference.goal(self.cfg.goal_mode, rotation=state[1])

    def _as_state(self, value: Any) -> tuple[torch.Tensor, ...]:
        if isinstance(value, tuple) and len(value) == 6:
            return value
        return self.unflatten(value)

    @staticmethod
    def _norm_std(normalizer: Any) -> torch.Tensor:
        fn = getattr(normalizer, "std", None)
        return fn() if callable(fn) else normalizer.std

    @staticmethod
    def _mean_tensor(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().mean()

    def _accumulate(
        self,
        metrics: dict[str, torch.Tensor],
        *,
        sample_counts: dict[str, int] | None = None,
    ) -> None:
        for name, value in metrics.items():
            count = None if sample_counts is None else sample_counts.get(name)
            if count is not None:
                value = value.detach().float().sum()
                self._pending_counts.setdefault(name, []).append(
                    torch.tensor(
                        count,
                        dtype=torch.float32,
                        device=value.device,
                    )
                )
            elif value.numel() != 1:
                value = self._mean_tensor(value)
            self._pending.setdefault(name, []).append(value.detach())

    def _update_running_norms(
        self,
        normalizers: tuple[Any, ...],
        samples: tuple[torch.Tensor, ...],
    ) -> None:
        if self.distributed is not None:
            self.distributed.update_running_norms(normalizers, samples)
            return
        for normalizer, sample in zip(normalizers, samples, strict=True):
            normalizer.update(sample)

    def _average_gradients(self, modules: nn.Module | tuple[nn.Module, ...]) -> None:
        if self.distributed is not None:
            self.distributed.average_gradients(_as_modules(modules))

    @staticmethod
    def _clip_gradient_norm(
        stage: str, modules: nn.Module | tuple[nn.Module, ...]
    ) -> torch.Tensor:
        """Check and clip one gradient norm once at the optimizer boundary.

        The world model and the perception encoder share an optimizer, so they
        must also share one clip: clipping them separately would let either
        branch escape the 1.0 global budget the paper sets per network.
        """
        parameters = [
            parameter
            for module in _as_modules(modules)
            for parameter in module.parameters()
        ]
        try:
            return nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        except RuntimeError as exc:
            raise FloatingPointError(f"{stage}: gradient norm is non-finite") from exc

    def _learning_rate(self, iteration: int) -> float:
        if iteration >= self.cfg.lr_decay_iters:
            return self.cfg.lr * self.cfg.lr_scale * self.cfg.lr_floor
        phase = 0.5 * (1.0 + math.cos(math.pi * iteration / self.cfg.lr_decay_iters))
        return (
            self.cfg.lr
            * self.cfg.lr_scale
            * (self.cfg.lr_floor + (1.0 - self.cfg.lr_floor) * phase)
        )

    def _set_lr(self, iteration: int) -> float:
        lr = self._learning_rate(iteration)
        self._current_lr = lr
        for optimizer in (self.opt_wm, self.opt_policy, self.opt_critic):
            for group in optimizer.param_groups:
                group["lr"] = lr
        return lr

    def _obs(
        self, state: tuple[torch.Tensor, ...], action: torch.Tensor, xi: torch.Tensor
    ):
        return self.mdp.obs(state, action, xi)

    def _relative_orientation(
        self, state, reference: ReferenceBatch, *, noise=False, generator=None
    ):
        value = relative_root_orientation(state[1], reference.now.R)
        if noise:
            draw = (
                torch.rand_like(value)
                if generator is None
                else torch.rand(
                    value.shape,
                    device=value.device,
                    dtype=value.dtype,
                    generator=generator,
                )
            )
            value = value + (2.0 * draw - 1.0) * RELATIVE_ORIENTATION_NOISE
        return value

    def _encode_perception(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode one batch of raw depth frames into the rollout latent.

        Runs under ``no_grad`` during collection: the policy acts on the latent,
        but the learning signal for the encoder comes from the supervised
        world-model update, not from this forward pass. Gradients through a
        closed-loop simulator step would be meaningless anyway.
        """
        if self.perception is None:
            raise RuntimeError("perception conditioning requested without a module")
        if "depth" not in observation:
            raise KeyError(
                "environment perception() must return a 'depth' frame of shape "
                f"[N, {DEPTH_SHAPE[0]}, {DEPTH_SHAPE[1]}] with invalid pixels at "
                f"INVALID_DEPTH; got keys {sorted(observation)}"
            )
        return self.perception.encode(observation["depth"], observation.get("valid"))

    def _log(self, *, final: bool = False) -> None:
        if not self._pending and not final:
            return
        local_sums = {
            name: torch.stack(values).sum() for name, values in self._pending.items()
        }
        local_counts = {
            name: (
                torch.stack(self._pending_counts[name]).sum()
                if name in self._pending_counts
                else torch.tensor(
                    len(values),
                    dtype=torch.float32,
                    device=local_sums[name].device,
                )
            )
            for name, values in self._pending.items()
        }
        self._pending.clear()
        self._pending_counts.clear()
        diagnostics = self.world_model_diagnostics()
        if self.sampler is not None:
            diagnostics.update(self.sampler.metrics())
            diagnostics.update(self.buffer.sampling_metrics)
        sharded = bool(getattr(self.sampler, "sharded", False))
        summed_diagnostics = set()
        if sharded:
            summed_diagnostics = {
                "Sampling/visits",
                "Sampling/failures",
                "Sampling/visited_bins",
            }
            summed_diagnostics.update(
                name
                for name in diagnostics
                if name.startswith("Sampling/")
                and name.endswith(("_global_candidates", "_supported_bins"))
            )
            diagnostics["Sampling/mastery_mean"] = (
                diagnostics["Sampling/mastery_mean"] * self.motion.num_bins
            )
            if self.distributed is not None and self.distributed.enabled:
                # One log-boundary collective obtains both extrema; max(x) = -min(-x).
                extrema = torch.stack(
                    (
                        diagnostics["Sampling/mastery_min"],
                        -diagnostics["Sampling/mastery_max"],
                    )
                )
                torch.distributed.all_reduce(extrema, op=torch.distributed.ReduceOp.MIN)
                diagnostics["Sampling/mastery_min"] = extrema[0]
                diagnostics["Sampling/mastery_max"] = -extrema[1]
        diagnostic_count = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=self.device,
        )
        if self.distributed is not None:
            gathered_names = self.distributed.all_gather_objects(tuple(local_sums))
            metric_names = sorted({name for names in gathered_names for name in names})
            metric_sums = [
                local_sums.get(
                    name,
                    torch.zeros((), dtype=torch.float32, device=self.device),
                )
                for name in metric_names
            ]
            metric_counts = [
                local_counts.get(
                    name,
                    torch.zeros((), dtype=torch.float32, device=self.device),
                )
                for name in metric_names
            ]
        else:
            metric_names = list(local_sums)
            metric_sums = list(local_sums.values())
            metric_counts = list(local_counts.values())
        metric_names.extend(diagnostics)
        metric_sums.extend(diagnostics.values())
        metric_counts.extend(
            diagnostic_count * self.motion.num_bins
            if sharded and name == "Sampling/mastery_mean"
            else diagnostic_count
            for name in diagnostics
        )
        if metric_sums:
            if self.distributed is not None:
                metric_sums, metric_counts = self.distributed.sum_and_count_tensors(
                    metric_sums,
                    metric_counts,
                )
            values = (
                torch.stack(
                    tuple(
                        value
                        if name in summed_diagnostics
                        else value / count.clamp_min(1.0)
                        for name, value, count in zip(
                            metric_names,
                            metric_sums,
                            metric_counts,
                            strict=True,
                        )
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
            metrics = dict(zip(metric_names, values, strict=True))
        else:
            metrics = {}
        metrics.update(
            {
                "Counts/transitions": float(self.transitions),
                "Counts/transitions_local": float(self.transitions),
                "Counts/transitions_global": float(self.global_transitions),
                "Counts/windows_local": float(self.windows),
                "Counts/windows_global": float(self.global_windows),
                "Counts/optimizersteps": float(self.optimizer_steps),
                "Counts/sim_steps_local": float(self.sim_steps),
                "Counts/sim_steps_global": float(self.global_sim_steps),
                "Training/world_size": float(self.world_size),
                "Training/envs_local": float(self.cfg.num_envs),
                "Training/envs_global": float(self.cfg.num_envs * self.world_size),
                "Training/wm_batch_local": float(self.cfg.wm_batch),
                "Training/wm_batch_global": float(self.cfg.wm_batch * self.world_size),
                "Training/wm_batches_per_iteration_local": float(self.cfg.wm_batches),
                "Training/wm_batches_per_iteration_global": float(
                    self.cfg.wm_batches * self.world_size
                ),
                "Training/policy_batch_local": float(self.cfg.policy_batch),
                "Training/policy_batch_global": float(
                    self.cfg.policy_batch * self.world_size
                ),
                "Training/lr_base": float(self.cfg.lr),
                "Training/lr_scale": float(self.cfg.lr_scale),
                "Training/lr_effective": float(self._current_lr),
                "Training/utd_per_vector_step": float(self.cfg.utd),
                "Training/wm_only_iters": float(self.cfg.wm_only_iters),
                "Training/policy_updates_enabled": float(
                    self.iteration > self.cfg.wm_only_iters
                ),
                "Training/collection_action_limit": _COLLECTION_ACTION_LIMIT,
                "Training/bound_bootstrap_value": float(self.cfg.bound_bootstrap_value),
                "Training/bootstrap_upper_bound": MAX_STEP_REWARD
                / (1.0 - self.cfg.gamma),
            }
        )
        metrics.update(self.profiler.drain())
        # The model timer measures one minibatch; report total time per update.
        for name in (
            "collection",
            "actor",
            "critic",
            "model",
            "pmg_prior",
            "pmg_execution",
            "pmg_prepare_per_control_step",
        ):
            key = f"Performance/{name}"
            if key in metrics:
                batches = self.cfg.wm_batches if name == "model" else 1
                metrics[f"Performance/{name}_s"] = metrics.pop(key) * batches / 1000
        now = time.monotonic()
        metrics["Performance/iterations_per_second"] = (
            self.iteration - self._last_log_iteration
        ) / max(now - self._last_log_time, 1e-9)
        self._last_log_time = now
        self._last_log_iteration = self.iteration
        if self.device.type == "cuda":
            metrics.update(self.cuda_cache.metrics())
            metrics["Memory/allocated_gib"] = (
                torch.cuda.memory_allocated(self.device) / 2**30
            )
            metrics["Memory/reserved_gib"] = (
                torch.cuda.memory_reserved(self.device) / 2**30
            )
            metrics["Memory/max_allocated_gib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**30
            )
        is_main = self.distributed is None or self.distributed.is_main
        if self.logger is not None and is_main:
            self.logger.log(metrics, self.iteration)
        if is_main:
            printable = " | ".join(
                f"{key} {value:.4g}"
                for key, value in metrics.items()
                if key.startswith(
                    ("Train/", "Model/loss", "Loss/", "Training/", "Counts/")
                )
            )
            print(f"iteration {self.iteration} | {printable}", flush=True)

    def train(self, *, refill_replay: bool = True) -> None:
        """Run from the current iteration, refilling replay after a resume."""
        if refill_replay:
            self.collect(self.cfg.warmup_steps)
            self.profiler.drain()
            self._pending.clear()
            self._pending_counts.clear()
        self._last_log_time = time.monotonic()
        self._last_log_iteration = self.iteration
        for iteration in range(self.iteration, self.cfg.iterations):
            self._set_lr(iteration)
            self.step_accumulator += 1.0 / self.cfg.utd
            collect_steps = int(self.step_accumulator)
            self.step_accumulator -= collect_steps
            self.collect(collect_steps)
            for _ in range(self.cfg.wm_batches):
                self._accumulate(self.update_world_model())
            if iteration >= self.cfg.wm_only_iters:
                self._accumulate(self.update_policy_and_critic())
            self.iteration = iteration + 1
            if (
                self.iteration % self.cfg.log_every == 0
                or self.iteration == self.cfg.iterations
            ):
                self._log(final=True)
            saved_this_iteration = bool(self.cfg.log_dir) and (
                self.iteration % self.cfg.save_every == 0
                or self.iteration == self.cfg.iterations
            )
            checkpoint = Path(self.cfg.log_dir) / f"model_{self.iteration}.pt"
            if saved_this_iteration:
                checkpoint = Path(self.cfg.log_dir) / f"model_{self.iteration}.pt"
                self.save_checkpoint(checkpoint)
                if self.distributed is None or self.distributed.is_main:
                    # One serialization, with an atomic latest-checkpoint alias.
                    latest = checkpoint.with_name("last.pt")
                    alias = latest.with_suffix(".link.tmp")
                    alias.unlink(missing_ok=True)
                    os.link(checkpoint, alias)
                    os.replace(alias, latest)
            if (
                self.evaluation_callback is not None
                and self.cfg.eval_every
                and (self.iteration % self.cfg.eval_every == 0)
            ):
                checkpoint = Path(self.cfg.log_dir) / f"model_{self.iteration}.pt"
                if not saved_this_iteration:
                    self.save_checkpoint(checkpoint)
                if self.distributed is None or self.distributed.is_main:
                    self.evaluation_callback(checkpoint)

    def run_healthcheck(self, target_iterations: int) -> None:
        """Run at least ten outer iterations and report actor/critic coverage."""
        if target_iterations < 10:
            raise ValueError("healthcheck requires at least 10 training iterations")
        if target_iterations <= self.cfg.wm_only_iters:
            raise ValueError(
                f"healthcheck target {target_iterations} must exceed "
                f"wm_only_iters={self.cfg.wm_only_iters} to exercise policy/critic updates"
            )
        start = self.iteration
        self.train(refill_replay=True)
        completed = self.iteration - start
        if completed < 10:
            raise RuntimeError(
                f"healthcheck completed only {completed} of the required 10 iterations"
            )
        policy_rounds = max(0, self.iteration - max(start, self.cfg.wm_only_iters))
        if self.distributed is None or self.distributed.is_main:
            print(
                f"healthcheck passed: {completed} outer iterations, "
                f"{policy_rounds} policy/critic update rounds, "
                f"{self.global_transitions} real transitions, "
                f"{self.optimizer_steps} optimizer steps",
                flush=True,
            )
