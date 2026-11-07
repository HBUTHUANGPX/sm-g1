"""Isaac-independent SuperMimic training library.

Paper optimization settings and RX-P2 adaptations live in :class:`TrainingConfig`. Application parsing,
simulation launch, and checkpoint selection live in the training script.
"""

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

from .action_bounds import ACTION_LIMIT as _COLLECTION_ACTION_LIMIT, clip_action
from .distributed import DistributedContext, default_lr_scale
from .mdp_legacy import MAX_STEP_REWARD, RELATIVE_ORIENTATION_NOISE, relative_root_orientation
from .motion import GOAL_PER_FRAME, GOAL_PER_FRAME_LEGACY
from .networks import WM_OUT
from .perception import (
    CLEAN_DEPTH_CLIP_M, DEPTH_SHAPE, HEIGHT_SCAN_DIM, HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE, height_scan_orientation, height_scan_shape,
)
from .perception_loss import cap_decoder_physics_gradients
from .recovery import recovery_provenance, remaining_protection
from .reference import ReferenceBatch
from .runtime_safety import PeriodicCudaCache


def _valid_depth(depth: torch.Tensor) -> torch.Tensor:
    """Recover the stereo validity mask from a stored depth frame.

    Invalid pixels are written as a negative sentinel, so a stored frame is
    self-describing at any range. A distance threshold would not work: the
    sensor's ``depth_limit`` is 8 m while the actor observation clips to 2 m, so
    a real surface at 3 m would be indistinguishable from a miss. Storing the
    mask separately would double the depth traffic through replay.
    """
    return depth >= 0.0


def _as_modules(modules: nn.Module | tuple[nn.Module, ...]) -> tuple[nn.Module, ...]:
    return modules if isinstance(modules, tuple) else (modules,)


_PAPER = "SuperMimic, arXiv:2610.28480"
_SOURCE_COMMIT = "27126bddd74381d164118ef83ee78edb58484a6b"


@dataclass
class TrainingConfig:
    """Paper optimization defaults plus explicit RX-P2 environment settings."""

    robot_name: str = "rx_p2"
    num_envs: int = 4096
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    hist: int = 10
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    policy_batch: int = 4096
    lr: float = 2e-4
    lr_scale: float = 1.0
    lr_decay_iters: int = 250_000
    lr_floor: float = 0.03
    gamma: float = 0.99
    gae_lambda: float = 0.95
    action_noise: float = 0.1
    actuator_type: str = "delayed_torque_speed"
    max_delay: int = 4
    # ERP camera range-query implementation.
    raycast_backend: str = "warp"
    # None inherits raycast_backend; set independently to mix camera and height.
    height_raycast_backend: str | None = None
    # "reference" (paper contract, 73-D per frame) or "state_relative" (60-D
    # legacy control arm; must not be used for production training).
    goal_mode: str = "reference"
    # Current Tracking-style feedback; legacy checkpoints explicitly disable it.
    actor_relative_orientation: bool = True
    # "residual" (Reproduced: p' = p + dt*v' + R_h*dp, q' = q + dt*qd' + dq) or
    # "increment" (plain per-step delta; control arm only).
    wm_target_mode: str = "residual"
    term_height: float = 0.20
    term_tilt: float = 0.8
    # Outer training iterations, not physics steps or individual optimizer steps.
    cuda_cache_clear_interval: int = 20
    # Recompute pure imagined forwards instead of retaining every MLP activation.
    actor_recompute: str = "policy"
    recovery_enabled: bool = False
    # Perception is an engineering extension: the paper contract is
    # proprioception plus the reference goal. When disabled, every perception
    # path is skipped and the checkpoint remains a pure-proprioception model.
    perception: bool = False
    perception_arch: str = "recurrent"
    perception_burn_in: int = 16
    perception_latent_dim: int = 64
    # Missing checkpoint fields retain the original yaw-aligned encoder width.
    height_scan_frame: str = "yaw"
    height_scan_size: tuple[float, float] = HEIGHT_SCAN_SIZE
    height_scan_resolution: float = HEIGHT_SCAN_RESOLUTION
    # Clean recurrent ablations; depth weight applies after the 8 m normalization.
    perception_height_weight: float = 1.0
    perception_depth_weight: float = 1.0
    perception_decoder_physics_ratio: float | None = None
    # Legacy latent-prediction checkpoints only. Recurrent losses use fixed
    # per-image metric normalization (1 m height, 8 m ERP, hit BCE).
    perception_latent_weight: float = 1.0
    perception_reconstruction_weight: float = 1.0
    perception_predicted_scan_weight: float = 1.0
    # Recurrent ERP uses an independent clean-height encoder in the critic.
    # The "latent" checkpoint format retains its original depth-latent critic.
    critic_perception: bool = False
    critic_epochs: int = 5
    critic_minibatches: int = 4
    warmup_steps: int = 64
    # Optional historical experiment; normal runs update all networks from step 1.
    wm_only_iters: int = 0
    bound_bootstrap_value: bool = True
    save_every: int = 2000
    log_every: int = 50
    eval_every: int = 4000
    seed: int = 0
    device: str = "cuda:0"
    motion_file: str = ""
    motion_sharding: str = "none"
    motion_scene_file: str = ""
    urdf: str | None = None
    robot_usd: str | None = None
    log_dir: str = ""
    log_project_name: str = "supermimic-rx-p2"
    run_name: str = ""
    smoke: bool = False
    healthcheck: bool = False
    healthcheck_loops: int = 12

    @property
    def height_scan_shape(self) -> tuple[int, int]:
        return height_scan_shape(self.height_scan_size, self.height_scan_resolution)

    @property
    def height_scan_dim(self) -> int:
        nx, ny = self.height_scan_shape
        return nx * ny

    @property
    def effective_height_raycast_backend(self) -> str:
        return self.height_raycast_backend or self.raycast_backend

    def validate(self) -> None:
        if self.motion_sharding not in ("none", "group"):
            raise ValueError("config.motion_sharding must be none or group")
        if self.raycast_backend not in ("warp", "ovrtx"):
            raise ValueError("config.raycast_backend must be warp or ovrtx")
        if self.height_raycast_backend not in (None, "warp", "ovrtx"):
            raise ValueError("config.height_raycast_backend must be warp, ovrtx, or None")
        if not isinstance(self.actor_relative_orientation, bool):
            raise ValueError("config.actor_relative_orientation must be bool")
        if not math.isfinite(self.term_height) or self.term_height <= 0.0:
            raise ValueError("config.term_height must be finite and positive")
        if type(self.cuda_cache_clear_interval) is not int or self.cuda_cache_clear_interval < 0:
            raise ValueError("config.cuda_cache_clear_interval must be a non-negative integer")
        if self.actor_recompute not in ("none", "policy", "full"):
            raise ValueError("config.actor_recompute must be none, policy, or full")
        height_scan_shape(self.height_scan_size, self.height_scan_resolution)
        if (
            tuple(self.height_scan_size) != HEIGHT_SCAN_SIZE
            or self.height_scan_resolution != HEIGHT_SCAN_RESOLUTION
        ) and not (self.perception and self.perception_arch == "recurrent"):
            raise ValueError("nondefault height_scan geometry requires recurrent perception")
        positive = (
            "num_envs",
            "iterations",
            "buffer_steps",
            "horizon",
            "goal_frames",
            "hist",
            "wm_window",
            "wm_batches",
            "wm_batch",
            "policy_batch",
            "lr_decay_iters",
            "critic_epochs",
            "critic_minibatches",
            "warmup_steps",
            "save_every",
            "log_every",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"config.{name} must be positive")
        if self.wm_only_iters < 0:
            raise ValueError("config.wm_only_iters must be non-negative")
        if not 0.0 < self.utd <= 1.0:
            raise ValueError("config.utd must be in (0, 1]")
        if (
            not math.isfinite(self.lr)
            or not math.isfinite(self.lr_scale)
            or self.lr <= 0.0
            or self.lr_scale <= 0.0
        ):
            raise ValueError(
                "config.lr and config.lr_scale must be finite and positive"
            )
        if not 0.0 <= self.gamma < 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma must be in [0, 1) and gae_lambda in [0, 1]")
        if self.buffer_steps < self.wm_window:
            raise ValueError("config.buffer_steps must cover config.wm_window")
        if self.goal_mode not in ("reference", "state_relative"):
            raise ValueError(
                f"config.goal_mode must be 'reference' or 'state_relative', "
                f"got {self.goal_mode!r}"
            )
        if self.wm_target_mode not in ("residual", "increment"):
            raise ValueError(
                "config.wm_target_mode must be 'residual' or 'increment', "
                f"got {self.wm_target_mode!r}"
            )
        if self.healthcheck and self.healthcheck_loops < 10:
            raise ValueError("--healthcheck-loops must be at least 10")
        if self.height_scan_frame not in ("yaw", "world"):
            raise ValueError("height_scan_frame must be yaw or world")
        if self.height_scan_frame == "world" and not (self.perception and self.perception_arch == "recurrent"):
            raise ValueError("world height_scan_frame requires recurrent perception")
        for name in ("perception_height_weight", "perception_depth_weight", "perception_decoder_physics_ratio"):
            value = getattr(self, name)
            if name == "perception_decoder_physics_ratio" and value is None:
                continue
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"config.{name} must be finite and non-negative")
        if (
            self.perception_height_weight != 1.0
            or self.perception_depth_weight != 1.0
            or self.perception_decoder_physics_ratio is not None
        ) and not (self.perception and self.perception_arch == "recurrent"):
            raise ValueError("perception loss ablations require clean recurrent perception")
        if self.perception:
            if self.perception_arch not in ("recurrent", "latent"):
                raise ValueError("unknown perception_arch")
            if self.perception_arch == "recurrent":
                if not self.critic_perception:
                    raise ValueError("recurrent ERP requires a clean-height critic")
                if self.perception_burn_in < 1:
                    raise ValueError("perception_burn_in must be positive")
            if self.perception_latent_dim <= 0:
                raise ValueError("config.perception_latent_dim must be positive")
            weights = (
                self.perception_latent_weight,
                self.perception_reconstruction_weight,
                self.perception_predicted_scan_weight,
            )
            if any(not math.isfinite(value) or value < 0.0 for value in weights):
                raise ValueError("perception loss weights must be finite and non-negative")
            if self.perception_reconstruction_weight <= 0.0:
                # A latent-only objective collapses toward a constant: nothing
                # else pins the 64-D code to terrain geometry.
                raise ValueError(
                    "config.perception_reconstruction_weight must be positive when "
                    "perception is enabled; a latent-only objective collapses"
                )
        elif self.critic_perception:
            raise ValueError("config.critic_perception requires config.perception")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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


class Trainer:
    """Pure collect/fit-world-model/actor-critic loop over injected objects."""

    # 73-D layout: p(3), 6-D rotation(6), v(3), w(3), q(29), qd(29).
    WM_ERROR_FLOOR = (
        (3e-3,) * 3
        + (5e-3,) * 6
        + (2e-2,) * 3
        + (5e-2,) * 3
    ) + (3e-3,) * 29 + (5e-2,) * 29

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
        optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer] | None = None,
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
                [cfg.perception_height_weight, cfg.perception_depth_weight,
                 ratio is not None, 0.0 if ratio is None else ratio],
                dtype=torch.float32, device=self.device,
            )
            self._perception_loss_metrics = dict(zip(
                ("Perception/height_weight", "Perception/depth_weight",
                 "Perception/decoder_physics_cap_enabled", "Perception/decoder_physics_ratio"),
                settings.unbind(), strict=True,
            ))
        self.world_size = (
            distributed.world_size if distributed is not None else 1
        )
        self.env = env
        self.buffer = buffer
        self.motion = mdp.motion
        self.reference = mdp.reference
        self.pmg = None
        self.sampler = getattr(env, "sampler", None)
        self.mdp = mdp
        for name, owner in (("env", env), ("buffer", buffer), ("mdp", mdp)):
            if bool(getattr(owner, "recovery_enabled", False)) != cfg.recovery_enabled:
                raise ValueError(f"{name}.recovery_enabled must match config.recovery_enabled")
        self.wm = world_model
        self.policy = policy
        if bool(getattr(policy, "relative_orientation_dim", 0)) != cfg.actor_relative_orientation:
            raise ValueError("policy relative_orientation input must match config.actor_relative_orientation")
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
            model_latent_dim = world_model.driver_dim if self.recurrent else world_model.latent_dim
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
            raise ValueError("a perception module was supplied but config.perception is False")
        self.perception = perception
        if self.recurrent:
            world_frame = cfg.height_scan_frame == "world"
            if getattr(perception, "world_frame", False) != world_frame or getattr(critic, "world_frame", False) != world_frame:
                raise ValueError("WM and critic height encoders must match config.height_scan_frame")
            for name, owner in (("perception", perception), ("critic", critic), ("buffer", buffer)):
                if getattr(owner, "scan_dim", HEIGHT_SCAN_DIM) != cfg.height_scan_dim:
                    raise ValueError(f"{name}.scan_dim must match config.height_scan_dim")
            env_cfg = getattr(env, "cfg", None)
            if env_cfg is not None and (
                tuple(getattr(env_cfg, "height_scan_size", HEIGHT_SCAN_SIZE)) != tuple(cfg.height_scan_size)
                or getattr(env_cfg, "height_scan_resolution", HEIGHT_SCAN_RESOLUTION) != cfg.height_scan_resolution
            ):
                raise ValueError("environment height_scan geometry must match training config")
        self.camera = None
        self.camera_geometry = None
        if self.recurrent:
            from .camera import CameraTransport

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
        self.provenance["recovery"] = recovery_provenance(cfg.recovery_enabled, cfg.term_height, cfg.term_tilt)
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
        self._floor = torch.tensor(self.WM_ERROR_FLOOR, dtype=torch.float32, device=self.device)
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
        if getattr(self.policy, "goal_norm", None) is not None and self.policy.goal_norm.mean.numel() != goal_dim:
            raise ValueError(
                f"policy goal dimension {self.policy.goal_norm.mean.numel()} does not match {per_frame}*goal_frames={goal_dim}"
            )

    def _goal(self, reference: ReferenceBatch, state: tuple[torch.Tensor, ...]) -> torch.Tensor:
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
            return nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
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

    def _obs(self, state: tuple[torch.Tensor, ...], action: torch.Tensor, xi: torch.Tensor):
        return self.mdp.obs(state, action, xi)

    def _relative_orientation(self, state, reference: ReferenceBatch, *, noise=False, generator=None):
        value = relative_root_orientation(state[1], reference.now.R)
        if noise:
            draw = (torch.rand_like(value) if generator is None else
                    torch.rand(value.shape, device=value.device, dtype=value.dtype, generator=generator))
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

    def _check_physics(self) -> None:
        if self.physics_health is not None:
            self.physics_health.check()

    def collect(self, steps: int) -> None:
        """Collect real simulator steps; all reductions stay on device until logging."""
        self._check_physics()
        if steps:
            self.cuda_cache.before_collection(self.iteration)
        with self.profiler.record("collection"):
            for _ in range(steps):
                state = self._as_state(self.env.state())
                xi = self.env.xi
                policy_observation = getattr(self.env, "policy_observation", None)
                if callable(policy_observation):
                    policy_obs = policy_observation()
                else:
                    policy_obs = self.mdp.add_noise(self._obs(state, self.env.a_prev, xi))
                fresh = self.env.ep_step == 0
                if self._obs_history is None:
                    self._obs_history = policy_obs.unsqueeze(1).expand(-1, self.cfg.hist, -1).clone()
                else:
                    shifted = torch.cat((self._obs_history[:, 1:], policy_obs.unsqueeze(1)), 1)
                    self._obs_history = torch.where(fresh[:, None, None], policy_obs.unsqueeze(1), shifted)
                if self.pmg is not None:
                    orientation_before_replan = (
                        self._relative_orientation(state, self.env.reference_batch())
                        if self.cfg.actor_relative_orientation else None
                    )
                    with self.profiler.record("pmg_prepare_per_control_step"):
                        self.pmg.prepare(self._obs_history, state)
                    decision_context = self.pmg.decision_context()
                plan_ids, phases = self.env.reference_address()
                reference = self.reference.sample(plan_ids, phases, self.cfg.goal_frames)
                goal = self._goal(reference, state)
                if self.pmg is not None:
                    q = reference.now.q[:, :getattr(self.motion, "action_dim", 27)]
                    excess = torch.maximum(torch.relu(q - self.mdp.kin.q_hi[:getattr(self.motion, "action_dim", 27)]),
                                           torch.relu(self.mdp.kin.q_lo[:getattr(self.motion, "action_dim", 27)] - q))
                    generated = plan_ids < 0
                    self._accumulate({
                        "PMG/reference_joint_limit_frame_fraction": (
                            (excess.gt(1e-4).any(-1) & generated).sum() / generated.sum().clamp_min(1)),
                        "PMG/reference_joint_limit_max_rad": torch.where(generated[:, None], excess, 0.).amax(),
                    })
                with torch.no_grad():
                    self._update_running_norms(
                        (self.policy.obs_norm, self.policy.goal_norm),
                        (policy_obs, goal),
                    )
                    latent = None
                    if self.perception is not None:
                        perception_frame = self.env.perception()
                        latent = (
                            self.policy.encode_perception(perception_frame["depth"])
                            if self.recurrent else self._encode_perception(perception_frame)
                        )
                        self._update_running_norms(
                            (self.policy.perception_norm,), (latent,)
                        )
                    policy_input = perception_frame["depth"] if self.recurrent else latent
                    if self.cfg.actor_relative_orientation:
                        observation = getattr(self.env, "policy_relative_orientation", None)
                        if callable(observation):
                            orientation = observation()
                            if self.pmg is not None:
                                delta = self._relative_orientation(state, reference) - orientation_before_replan
                                orientation = torch.where(self.pmg.enabled[:, None], orientation + delta, orientation)
                        else:
                            orientation = self._relative_orientation(state, reference, noise=True)
                        action = self.policy(self._obs_history, goal, policy_input, orientation)
                    else:
                        action = self.policy(self._obs_history, goal, policy_input)
                    noisy_action = action + self.cfg.action_noise * torch.randn_like(action)
                    # Noise precedes the shared execution bound; Policy.forward stays raw.
                    action = clip_action(noisy_action)
                    self._accumulate({
                        "Collection/action_clip_fraction": (noisy_action.abs() > _COLLECTION_ACTION_LIMIT).float().mean(),
                        "Collection/raw_action_abs_max_mean": noisy_action.abs().amax(),
                    })
                    if self.pmg is not None:
                        plan_ids = plan_ids.clone()
                        self.reference.retain(plan_ids)
                    try:
                        executed = self.env.step(action)
                        self._check_physics()
                        if self.pmg is not None:
                            executed["policy_obs"] = self._obs_history[:, -1]
                            executed.update(decision_context)
                        self.buffer.add(executed)
                    finally:
                        if self.pmg is not None:
                            self.reference.release(plan_ids)
                            self.pmg.release_decision_context()
                    state0 = self._as_state(executed["s"])
                    state1 = self._as_state(executed["s1"])
                    latent1 = None
                    if self.recurrent:
                        latent = self._encode_clean_perception(
                            state0, executed["scan"], executed["hit"], executed["clean_depth"]
                        )
                    elif self.perception is not None:
                        # step() may already have reset the environment. Only
                        # its explicit endpoint belongs to the physical s1.
                        latent1 = self._encode_perception({"depth": executed["depth1"]})
                    self._update_running_norms(
                        (self.wm.in_norm, self.wm.out_norm),
                        (
                            self.wm.inputs(
                                state0,
                                executed["a"],
                                executed["a_prev"],
                                self.normalize_xi(executed["xi"]),
                                latent,
                            ),
                            self.wm.targets(state0, state1) if self.recurrent
                            else self.wm.targets(state0, state1, latent, latent1),
                        ),
                    )
                self.sim_steps += self.cfg.num_envs
                self.transitions += self.cfg.num_envs
                self.global_sim_steps += self.cfg.num_envs * self.world_size
                self.global_transitions += self.cfg.num_envs * self.world_size
                self._accumulate(
                    {
                        "Train/reward": self._mean_tensor(executed["r"]) / 0.1,
                        "Tracking/survival": (1.0 - executed.get("done", executed["term"]).float()).mean(),
                        "Train/termination": executed["term"].float().mean(),
                        "Tracking/root_height_error_m": (
                            reference.now.p[:, 2] - state0[0][:, 2]
                        ).abs().mean(),
                        "Tracking/gravity_z_error": (
                            reference.now.R[:, 2, 2] - state0[1][:, 2, 2]
                        ).abs().mean(),
                    }
                )
                if "ep_len" in executed:
                    self._accumulate(
                        {"Train/episode_length": executed["ep_len"]},
                        sample_counts={
                            "Train/episode_length": executed["ep_len"].numel()
                        },
                    )
                if "ep_ret_norm" in executed:
                    self._accumulate(
                        {"Train/reward_fraction_actual_length": executed["ep_ret_norm"]},
                        sample_counts={
                            "Train/reward_fraction_actual_length": executed["ep_ret_norm"].numel()
                        },
                    )
                if "terms" in executed:
                    for name, value in executed["terms"].items():
                        self._accumulate({f"Reward/{name}": value})
                    for name in ("e_body_pos", "e_body_ori", "e_root_pos", "action_rate", "joint_limit"):
                        if name in executed["terms"]:
                            self._accumulate({f"Tracking/{name}": executed["terms"][name]})

    def _encode_clean_perception(self, state, scan, hit, depth):
        if getattr(self.perception, "world_frame", False):
            return self.perception.encode(scan, hit, depth, height_scan_orientation(state[1]))
        return self.perception.encode(scan, hit, depth)

    @torch.no_grad()
    def _burnin_hidden(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
        """Recompute recurrent context with current weights, within one episode.

        The finite prefix is truncated BPTT context, not a cached hidden state
        from older model weights. Padded rows leave the zero reset state intact.
        """
        batch, length = history["mask"].shape
        hidden = self.perception.initial_hidden(batch, self.device)
        if length == 0:
            return hidden
        # Recorded frames are independent inputs to the encoders and WM. Batch
        # those computations; only the GRU recurrence must run in time order.
        historical_state = self._as_state(history["s"].flatten(0, 1))
        features = self._encode_clean_perception(
            historical_state, history["scan"].flatten(0, 1), history["hit"].flatten(0, 1),
            history["clean_depth"].flatten(0, 1).float(),
        )
        _, driver = self.wm.step(
            historical_state, history["a"].flatten(0, 1),
            history["a_prev"].flatten(0, 1), self.normalize_xi(history["xi"].flatten(0, 1)), features,
        )
        features = features.reshape(batch, length, -1)
        driver = driver.reshape(batch, length, -1)
        for step in range(length):
            next_hidden = self.perception.advance_hidden(features[:, step], driver[:, step], hidden)
            hidden = torch.where(history["mask"][:, step, None], next_hidden, hidden)
        return hidden

    def _clean_prediction(self, state, action, previous, xi, scan, hit, depth, hidden):
        features = self._encode_clean_perception(state, scan, hit, depth)
        state, driver = self.wm.step(state, action, previous, xi, features)
        hidden, scan, logits, depth = self.perception.step(features, driver, hidden, _valid_depth(depth))
        return state, hidden, scan, logits, depth

    @staticmethod
    def _perception_errors(scan, logits, depth, target_scan, target_hit, target_depth):
        # Pixel/cell means prevent resolution from scaling the model objective.
        height_error = torch.where(target_hit, scan - target_scan.float(), 0.0)
        height_count = target_hit.sum(-1).clamp_min(1)
        depth_valid = _valid_depth(target_depth)
        depth_error = torch.where(depth_valid, depth - target_depth.float(), 0.0)
        depth_count = depth_valid.flatten(1).sum(-1).clamp_min(1)
        height_mse = height_error.square().sum(-1) / height_count
        depth_mse = depth_error.square().flatten(1).sum(-1) / depth_count
        hit_bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target_hit.to(logits.dtype), reduction="none"
        ).mean(-1)
        return height_mse, depth_mse, hit_bce

    def _clean_world_model_rollout(self, sample, window):
        state = self._as_state(sample["s"][:, 0])
        hidden = self._burnin_hidden(sample["burnin"])
        scan, hit = sample["scan"][:, 0].float(), sample["hit"][:, 0]
        depth = sample["clean_depth"][:, 0].float()
        losses, physical, normalized, sensory = [], [], [], []
        reconstruction_losses = []
        # Prediction-only runners predate the optional loss configuration.
        config = getattr(self, "cfg", None)
        height_weight = getattr(config, "perception_height_weight", 1.0)
        depth_weight = getattr(config, "perception_depth_weight", 1.0)
        cap_ratio = getattr(config, "perception_decoder_physics_ratio", None)
        std = torch.maximum(self._norm_std(self.wm.out_norm), self._floor)
        for step in range(window):
            state, hidden, scan, logits, depth = self._clean_prediction(
                state, sample["a"][:, step], sample["a_prev"][:, step],
                self.normalize_xi(sample["xi"][:, step]), scan, hit, depth, hidden,
            )
            hit = logits.sigmoid()
            mask = sample["mask"][:, step]
            error = self.wm.state_error(state, self._as_state(sample["s1"][:, step]))
            height_mse, depth_mse, hit_bce = self._perception_errors(
                scan, logits, depth, sample["scan1"][:, step],
                sample["hit1"][:, step], sample["clean_depth1"][:, step],
            )
            # Physical SSE is unchanged. Sensor losses have fixed metric scales:
            # height / 1 m, ERP / 8 m, and hit BCE; all averaged per image.
            physics = error.square().sum(-1)
            losses.append(((
                physics + height_weight * height_mse
                + depth_weight * (depth_mse / CLEAN_DEPTH_CLIP_M**2)
                + hit_bce
            ) * mask).sum())
            if cap_ratio is not None:
                reconstruction_losses.append(((height_weight * height_mse
                    + depth_weight * (depth_mse / CLEAN_DEPTH_CLIP_M**2) + hit_bce) * mask).sum())
            physical.append((error.square().mean(-1) * mask).sum())
            normalized.append(((error / std).square().mean(-1) * mask).sum())
            sensory.append(torch.stack(((height_mse * mask).sum(), (depth_mse * mask).sum(), (hit_bce * mask).sum())))
        means = torch.stack(sensory).sum(0) / sample["mask"].sum().clamp_min(1)
        return {
            "state": state, "latent": None, "losses": losses,
            "reconstruction_loss": torch.stack(reconstruction_losses).sum() if reconstruction_losses else None,
            "physical_errors": physical, "normalized_errors": normalized, "latent_errors": [],
            "scan_metrics": {
                "Perception/clean_height_rmse_m": means[0].sqrt().detach(),
                "Perception/clean_depth_rmse_m": means[1].sqrt().detach(),
                "Perception/height_hit_bce": means[2].detach(),
                "Loss/height_prediction": means[0].detach(),
                "Loss/depth_prediction": (means[1] / CLEAN_DEPTH_CLIP_M**2).detach(),
                "Loss/height_hit": means[2].detach(),
                **self._perception_loss_metrics,
            },
        }

    def _world_model_rollout(self, sample: dict[str, torch.Tensor], window: int):
        if self.recurrent:
            return self._clean_world_model_rollout(sample, window)
        state = self._as_state(sample["s"][:, 0])
        latent = self._latent_at(sample, 0) if self.perception is not None else None
        losses: list[torch.Tensor] = []
        normalized_errors: list[torch.Tensor] = []
        physical_errors: list[torch.Tensor] = []
        latent_errors: list[torch.Tensor] = []
        scan_metrics: dict[str, torch.Tensor] = {}
        std = torch.maximum(self._norm_std(self.wm.out_norm), self._floor)
        latent_std = std[..., WM_OUT:] if self.perception is not None else None
        for step in range(window):
            mask = sample["mask"][:, step]
            if self.perception is None:
                state = self.wm.step(
                    state,
                    sample["a"][:, step],
                    sample["a_prev"][:, step],
                    self.normalize_xi(sample["xi"][:, step]),
                )
                error = self.wm.state_error(state, self._as_state(sample["s1"][:, step]))
            else:
                state, latent = self.wm.step(
                    state,
                    sample["a"][:, step],
                    sample["a_prev"][:, step],
                    self.normalize_xi(sample["xi"][:, step]),
                    latent,
                )
                target = self._as_state(sample["s1"][:, step])
                error = self.wm.state_error(state, target)
                # Encode the pre-reset endpoint with current weights. The
                # detached target does not pull the encoder toward predictions.
                with torch.no_grad():
                    target_latent = self._latent_at(sample, step, next_state=True)
                latent_error = latent - target_latent
                losses.append(
                    latent_error.square().mean(-1)
                    .mul(self.cfg.perception_latent_weight)
                    .mul(mask)
                    .sum()
                )
                latent_errors.append(
                    (latent_error / latent_std).square().mean(-1).mul(mask).sum()
                )
            normalized = (error[..., :WM_OUT] / std[..., :WM_OUT]) ** 2
            # Local raw-state SSE choice. Eq. (1) leaves state scaling unspecified;
            # the reference implementation instead uses inverse-variance weights.
            losses.append((error[..., :WM_OUT].square().sum(-1) * mask).sum())
            normalized_errors.append((normalized.mean(-1) * mask).sum())
            physical_errors.append((error[..., :WM_OUT].square().mean(-1) * mask).sum())
        if self.perception is not None:
            # Two reconstruction terms, each evaluated once per window:
            #
            # * the recorded first frame pins the encoder's latent to terrain
            #   geometry, which is what stops a latent-only objective from
            #   collapsing toward a constant;
            # * the rolled-out last latent is decoded to the scan at t+k, so the
            #   predictor is also scored on the metric quantity the actor will
            #   effectively see during imagination.
            first_mask = sample["mask"][:, 0]
            losses.append(
                self._reconstruction_loss(
                    self._latent_at(sample, 0), sample, 0, first_mask
                )
            )
            scan_metrics = self._scan_metrics(
                latent, sample, window - 1, sample["mask"][:, window - 1], next_state=True
            )
            losses.append(
                scan_metrics.pop("Perception/predicted_sse")
                * self.cfg.perception_predicted_scan_weight
            )
        return {
            "state": state,
            "latent": latent,
            "losses": losses,
            "normalized_errors": normalized_errors,
            "physical_errors": physical_errors,
            "latent_errors": latent_errors,
            "scan_metrics": scan_metrics,
        }

    def _latent_at(
        self, sample: dict[str, torch.Tensor], index: int, *, next_state: bool = False
    ) -> torch.Tensor:
        frame = sample["depth1" if next_state else "depth"][:, index]
        return self.perception.encode(frame, _valid_depth(frame))

    def _reconstruction_loss(
        self,
        latent: torch.Tensor,
        sample: dict[str, torch.Tensor],
        index: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the recorded height scan from a latent.

        The scan is the encoder's supervision target in metric space; the
        decoder is scored with the same alive mask as the dynamics so a window
        never trains on frames past a reset.
        """
        scan = sample["scan"][:, index].float()
        hit = sample["hit"][:, index]
        predicted_scan, hit_logit = self.perception.decode(latent)
        observed = hit & mask[:, None].bool()
        height_error = torch.where(observed, predicted_scan - scan, 0.0)
        reconstruction = height_error.square().sum(-1)
        bce = nn.functional.binary_cross_entropy_with_logits(
            hit_logit, hit.to(scan.dtype), reduction="none"
        ).mean(-1)
        return ((reconstruction + bce) * self.cfg.perception_reconstruction_weight).mul(mask).sum()

    def _scan_metrics(
        self,
        latent: torch.Tensor,
        sample: dict[str, torch.Tensor],
        index: int,
        mask: torch.Tensor,
        *,
        next_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Report the predicted latent's decoded scan error in metres.

        Two numbers, kept separate on purpose: the latent L2 says whether the
        rollout state stayed close, and the decoded MAE says whether the latent
        still carries the terrain. A shrinking latent looks healthy under the
        first and fails the second.
        """
        scan = sample["scan1" if next_state else "scan"][:, index].float()
        hit = sample["hit1" if next_state else "hit"][:, index]
        predicted_scan, hit_logit = self.perception.decode(latent)
        denominator = mask.sum().clamp_min(1.0)
        observed = hit & mask[:, None].bool()
        height_error = torch.where(observed, predicted_scan - scan, 0.0)
        # Losses are batch sums here; update_world_model divides once by B.
        sse = (height_error.square().sum(-1) * mask).sum()
        hit_count = (hit * mask[:, None]).sum().clamp_min(1.0)
        mae = (height_error.abs() * mask[:, None]).sum() / hit_count
        bce = (
            nn.functional.binary_cross_entropy_with_logits(
                hit_logit, hit.to(scan.dtype), reduction="none"
            ).mean(-1)
            * mask
        ).sum() / denominator
        return {
            "Perception/predicted_scan_mae_m": mae.detach(),
            "Perception/predicted_hit_bce": bce.detach(),
            "Perception/predicted_sse": sse,
        }

    def update_world_model(self) -> dict[str, torch.Tensor]:
        """Run one of the ten independently sampled world-model Adam updates."""
        self._check_physics()
        with self.profiler.record("model"):
            sample = self.buffer.sample_windows(
                self.cfg.wm_batch, self.cfg.wm_window,
                **({"motion": self.motion, "bin_probabilities": self.sampler.wm_probabilities}
                   if self.sampler is not None else {}),
            )
            if self.sampler is not None:
                # Replay caps the actual rank-local proposal. Importance
                # weights retain the global bin objective under DDP averaging.
                sample = {**sample, "mask": sample["mask"] * sample["sampling_weight"][:, None]}
            rollout = self._world_model_rollout(sample, self.cfg.wm_window)
            losses = rollout["losses"]
            normalized_errors = rollout["normalized_errors"]
            physical_errors = rollout["physical_errors"]
            batch = self.cfg.wm_batch
            first_mask = sample["mask"][:, 0]
            last_mask = sample["mask"][:, -1]
            # Eq. 1 is feature-SUM across the window, MEAN over sampled windows.
            # The replay window length is not part of the loss denominator.
            loss = torch.stack(losses).sum() / batch
            predicted = self._wm_modules()
            self.opt_wm.zero_grad(set_to_none=True)
            cap_ratio = self.cfg.perception_decoder_physics_ratio if self.recurrent else None
            cap_metrics = {}
            if cap_ratio is not None:
                decoders = {
                    "height": self.perception.height_decoder,
                    "depth": self.perception.depth_decoder,
                }
                # Directly compute the smaller reconstruction contribution;
                # total-minus-physics would erase it when physics dominates.
                reconstruction_gradients = torch.autograd.grad(
                    rollout["reconstruction_loss"] / batch,
                    tuple(p for module in decoders.values() for p in module.parameters()),
                    retain_graph=True, allow_unused=True,
                )
            loss.backward()
            self._average_gradients(predicted)
            if cap_ratio is not None:
                cap_metrics = cap_decoder_physics_gradients(
                    decoders, reconstruction_gradients, cap_ratio, distributed=self.distributed,
                )
            grad_norm = self._clip_gradient_norm("world model update", predicted)
            self.opt_wm.step()
            self.optimizer_steps += 1
            self.windows += batch
            self.global_windows += batch * self.world_size
            rmse_1 = torch.sqrt(physical_errors[0] / first_mask.sum().clamp_min(1))
            rmse_last = torch.sqrt(physical_errors[-1] / last_mask.sum().clamp_min(1))
            normalized_1 = torch.sqrt(
                normalized_errors[0] / first_mask.sum().clamp_min(1)
            )
            normalized_last = torch.sqrt(
                normalized_errors[-1] / last_mask.sum().clamp_min(1)
            )
            return {
                "Model/loss": loss.detach(),
                "Model/physical_rmse_1": rmse_1.detach(),
                f"Model/physical_rmse_{self.cfg.wm_window}": rmse_last.detach(),
                "Model/running_std_normalized_rmse_1": normalized_1.detach(),
                f"Model/running_std_normalized_rmse_{self.cfg.wm_window}": normalized_last.detach(),
                "Gradient/world_model": grad_norm.detach(),
                "Model/mean_running_std_normalized_window_error": (torch.stack(normalized_errors).sum() / sample["mask"].sum().clamp_min(1)).detach(),
                "Model/mean_physical_mse": (torch.stack(physical_errors).sum() / sample["mask"].sum().clamp_min(1)).detach(),
                **rollout["scan_metrics"],
                **cap_metrics,
                **(
                    {
                        "Perception/latent_normalized_rmse": torch.sqrt(
                            torch.stack(rollout["latent_errors"]).sum()
                            / sample["mask"].sum().clamp_min(1)
                        ).detach()
                    }
                    if rollout["latent_errors"]
                    else {}
                ),
            }

    def _critic_input(
        self,
        state: tuple[torch.Tensor, ...],
        reference: ReferenceBatch,
        xi_n: torch.Tensor,
        goal: torch.Tensor,
        latent: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        remaining: torch.Tensor | None = None,
        body_state: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        context = {"remaining": remaining, "body_state": body_state} if self.cfg.recovery_enabled else {}
        parts = [self.mdp.critic_features(state, reference, xi_n, **context), goal]
        if self.recurrent:
            scan, hit = latent
            parts.extend((scan, hit.to(scan.dtype)))
            if self.critic.world_frame:
                parts.append(height_scan_orientation(state[1]))
        elif self.cfg.critic_perception:
            if latent is None or latent.shape[-1] != self.cfg.perception_latent_dim:
                raise ValueError(
                    "critic_perception requires a latent with width "
                    f"{self.cfg.perception_latent_dim}"
                )
            parts.append(latent)
        result = torch.cat(parts, -1)
        expected = getattr(self.critic, "in_dim", result.shape[-1])
        if result.shape[-1] != expected:
            raise ValueError(
                f"critic input width {result.shape[-1]} does not match Critic({expected}); "
                f"critic_perception={self.cfg.critic_perception}"
            )
        return result

    def _wm_modules(self) -> tuple[nn.Module, ...]:
        """Modules sharing the world-model optimizer and its gradient budget."""
        if self.perception is None:
            return (self.wm,)
        return (self.wm, self.perception)

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(value)

    def _bootstrap_value(self, value: torch.Tensor) -> torch.Tensor:
        """Optional A/B bound on targets; raw predictions still train under MSE."""
        if not self.cfg.bound_bootstrap_value:
            return value
        upper = MAX_STEP_REWARD / (1.0 - self.cfg.gamma)
        # A finite reward bound must not hide NaN/Inf in a failed critic.
        return torch.where(torch.isfinite(value), value.clamp_max(upper), value)

    def update_policy_and_critic(self) -> dict[str, torch.Tensor]:
        """One 4096-rollout BPTT actor update followed by five critic epochs."""
        self._check_physics()
        with self.profiler.record("actor"):
            seed, history = self.buffer.sample_starts(
                self.cfg.policy_batch, self.cfg.hist, self.cfg.horizon, self.motion,
                **({"bin_probabilities": self.sampler.actor_probabilities}
                   if self.sampler is not None else {}),
            )
            state0 = self._as_state(seed["s"])
            xi = seed["xi"]
            xi_n = self.normalize_xi(xi)
            plan_ids, phases0 = seed["ref_id"], seed["ref_phase"]
            reference0 = self.reference.sample(plan_ids, phases0, self.cfg.goal_frames)
            reference = reference0
            action_prev = seed["a_prev"]
            state = state0
            goal0 = self._goal(reference0, state0)
            goal = goal0
            with torch.no_grad():
                obs_history = self._obs(
                    self._as_state(history["s"]), history["a_prev"], history["xi"]
                )
            obs_history = self.mdp.add_noise(obs_history)
            states = []
            actions = []
            action_clip_counts = []
            action_maxima = []
            previous_actions = []
            goals = []
            # The latent entering imagination is the one encoded from the seed's
            # recorded frame. From there the world model predicts it, so the
            # actor never sees a real depth frame the rollout could not have.
            latent = None
            latents = []
            height_history = []
            frozen_modules = (*self._wm_modules(), self.critic)
            frozen_state = [
                (module.training, [parameter.requires_grad for parameter in module.parameters()])
                for module in frozen_modules
            ]
            for module in frozen_modules:
                module.eval()
                self._set_requires_grad(module, False)
            try:
                policy_forward = self.policy
                model_step = self._clean_prediction if self.recurrent else self.wm.step
                # These calls are deterministic and only read normalization state.
                # Noise, camera queues and geometry remain outside recomputation.
                if self.cfg.actor_recompute != "none":
                    policy_forward = partial(checkpoint, policy_forward, use_reentrant=False,
                                             preserve_rng_state=False)
                if self.cfg.actor_recompute == "full":
                    # The nested depth-decoder checkpoint creates factory tensors.
                    # Its forward and backward recomputation need the same device.
                    context = (
                        {"context_fn": lambda: (torch.device(self.device), torch.device(self.device))}
                        if self.recurrent else {}
                    )
                    model_step = partial(checkpoint, model_step, use_reentrant=False,
                                         preserve_rng_state=False, **context)
                if self.recurrent:
                    hidden = self._burnin_hidden(seed["burnin"])
                    scan, hit = seed["scan"].float(), seed["hit"]
                    depth = seed["clean_depth"].float()
                    camera_state = self.camera.seed(seed["camera_frames"].float(), seed["camera_lag"].long())
                    latent = self.camera.read(camera_state)
                    height_history.append((scan, hit))
                    # Imagined episodes do not reset. Capture rows repeat every
                    # camera period; compact five phases once instead of H times.
                    capture_rows = tuple(
                        ((seed["ep_step"] + offset) % self.camera.capture_steps == 0).nonzero().flatten()
                        for offset in range(self.camera.capture_steps)
                    )
                elif self.perception is not None:
                    with torch.no_grad():
                        latent = self.perception.encode(seed["depth"], _valid_depth(seed["depth"]))
                if self.cfg.critic_perception and not self.recurrent:
                    latents.append(latent)
                for step in range(self.cfg.horizon):
                    next_reference = self.reference.sample(plan_ids, phases0 + step + 1, self.cfg.goal_frames)
                    previous_actions.append(action_prev)
                    if self.cfg.actor_relative_orientation:
                        # Use this pre-action predicted state/frame. Pass the tensor
                        # explicitly so recomputation cannot read a later reference.
                        orientation = self._relative_orientation(state, reference, noise=True)
                        raw_action = policy_forward(obs_history, goal, latent, orientation)
                    else:
                        raw_action = policy_forward(obs_history, goal, latent)
                    action = clip_action(raw_action)
                    raw_magnitude = raw_action.detach().abs()
                    action_clip_counts.append((raw_magnitude > _COLLECTION_ACTION_LIMIT).sum())
                    action_maxima.append(raw_magnitude.amax())
                    if self.recurrent:
                        state, hidden, scan, logits, depth = model_step(
                            state, action, action_prev, xi_n, scan, hit, depth, hidden
                        )
                        hit = logits.sigmoid()
                        camera_state, latent = self.camera.advance(
                            camera_state, depth, seed["ep_step"] + step + 1,
                            capture_ids=capture_rows[(step + 1) % self.camera.capture_steps],
                            occlusion_query=lambda clean, ids: self.camera_geometry.visibility(
                                tuple(value[ids] for value in state), motion_ids=next_reference.scene_id[ids]
                            ),
                        )
                        height_history.append((scan, hit))
                    elif self.perception is None:
                        state = model_step(state, action, action_prev, xi_n)
                    else:
                        state, latent = model_step(
                            state, action, action_prev, xi_n, latent
                        )
                    if self.cfg.critic_perception and not self.recurrent:
                        latents.append(latent)
                    reference = next_reference
                    obs = self.mdp.add_noise(self._obs(state, action, xi))
                    obs_history = torch.cat((obs_history[:, 1:], obs.unsqueeze(1)), 1)
                    goal = self._goal(reference, state)
                    states.append(state)
                    actions.append(action)
                    goals.append(goal)
                    action_prev = action

                batch = self.cfg.policy_batch
                horizon = self.cfg.horizon
                flat = lambda values: torch.stack(values).flatten(0, 1)  # noqa: E731
                states_flat = tuple(flat([item[i] for item in states]) for i in range(6))
                future_phases = phases0[None] + torch.arange(1, horizon + 1, device=self.device)[:, None]
                rollout_ids = plan_ids[None].expand_as(future_phases)
                rollout_reference = self.reference.sample(rollout_ids.flatten(), future_phases.flatten(), bodies=True)
                actions_flat = flat(actions)
                previous_flat = flat(previous_actions)
                body_context, clock_context, tail_context = {}, {}, {}
                if self.cfg.recovery_enabled:
                    # Share the post-state FK across reward, hazard and both
                    # critic paths. The WM never predicts an episode clock.
                    bodies = self.mdp.kin.forward(*states_flat)
                    body_context = {"body_state": bodies}
                    ages = seed["ep_step"][None] + torch.arange(horizon, device=self.device)[:, None]
                    roles = seed["recovery_role"][None].expand_as(ages)
                    clock_context = {"recovery_role": roles.flatten(), "ep_step": ages.flatten()}
                    tail_remaining = remaining_protection(seed["recovery_role"], seed["ep_step"] + horizon)
                    tail_context = {
                        "remaining": tail_remaining,
                        "body_state": tuple(value.reshape(horizon, batch, *value.shape[1:])[-1] for value in bodies),
                    }
                rewards = self.mdp.reward(states_flat, rollout_reference, actions_flat, previous_flat, **body_context).view(horizon, batch)
                termination = self.mdp.soft_termination(states_flat, rollout_reference, **body_context, **clock_context).view(horizon, batch)
                source_reference = self.reference.sample(rollout_ids, future_phases - 1)
                valid_reference = reference0.valid[None] & source_reference.valid
                clip_finished = rollout_reference.expiry.episode_end.view(horizon, batch)
                rewards = torch.where(valid_reference, rewards, 0.0)
                termination = torch.where(clip_finished, 1.0, termination)
                alive = torch.cumprod(1.0 - termination, 0)
                survival_at_step = torch.cat((torch.ones(1, batch, device=self.device), alive[:-1]), 0)
                discount = self.cfg.gamma ** torch.arange(
                    horizon, device=self.device, dtype=torch.float32
                ).unsqueeze(-1)
                imagined_return = (discount * survival_at_step * rewards).sum(0)
                survival = alive[-1]
                tail_reference = self.reference.sample(plan_ids, phases0 + horizon, bodies=True)
                final_critic = self.critic(
                    self._critic_input(state, tail_reference, xi_n, goal, (scan, hit) if self.recurrent else latent, **tail_context)
                )
                reward_return = imagined_return
                bootstrap_return = (self.cfg.gamma**horizon) * survival * self._bootstrap_value(final_critic)
                imagined_return = reward_return + bootstrap_return
                replay_weight = seed["sampling_weight"] if self.sampler is not None else 1.0
                actor_loss = -(imagined_return * replay_weight).mean()
                self.opt_policy.zero_grad(set_to_none=True)
                actor_loss.backward()
                self._average_gradients(self.policy)
                actor_grad = self._clip_gradient_norm("actor update", self.policy)
                nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
                self.opt_policy.step()
                self.optimizer_steps += 1

                reward_sum = (survival_at_step * rewards).detach().sum(0)
                tail_over_bound = (
                    (final_critic.detach() > MAX_STEP_REWARD / (1.0 - self.cfg.gamma))
                    & torch.isfinite(final_critic.detach())
                ).float().mean()
                actor_metrics = {
                    "Loss/actor": actor_loss.detach(),
                    "Gradient/actor": actor_grad.detach(),
                    "Tracking/imagined_return": (reward_sum / horizon / 0.1).mean().detach(),
                    "Tracking/imagined_survival": survival.mean().detach(),
                    "Actor/reward_return": (reward_return * replay_weight).mean().detach(),
                    "Actor/bootstrap_return": (bootstrap_return * replay_weight).mean().detach(),
                    "Critic/raw_tail_value_max_mean": final_critic.detach().amax(),
                    "Critic/raw_tail_over_bound_fraction": tail_over_bound,
                    "Critic/bootstrap_bound_fraction": (
                        tail_over_bound
                        if self.cfg.bound_bootstrap_value else final_critic.new_zeros(())
                    ),
                    "Imagination/action_clip_fraction": (
                        torch.stack(action_clip_counts).sum() / (horizon * batch * actions[0].shape[-1])
                    ),
                    "Imagination/raw_action_abs_max_mean": torch.stack(action_maxima).amax(),
                }
            finally:
                for module, (training, gradients) in zip(frozen_modules, frozen_state, strict=True):
                    module.train(training)
                    for parameter, requires_grad in zip(module.parameters(), gradients, strict=True):
                        parameter.requires_grad_(requires_grad)

        with self.profiler.record("critic"):
            with torch.no_grad():
                states_with_seed = tuple(
                    torch.cat((state0[i].unsqueeze(0), torch.stack([item[i].detach() for item in states])), 0)
                    for i in range(6)
                )
                phases_with_seed = phases0[None] + torch.arange(horizon + 1, device=self.device)[:, None]
                ids_with_seed = plan_ids[None].expand_as(phases_with_seed)
                critic_reference = self.reference.sample(ids_with_seed.flatten(), phases_with_seed.flatten(), bodies=True)
                goals_with_seed = torch.cat((goal0.unsqueeze(0), torch.stack(goals)), 0)
                xi_with_seed = xi_n.unsqueeze(0).expand(self.cfg.horizon + 1, -1, -1)
                flat_states = tuple(
                    value.flatten(0, 1) for value in states_with_seed
                )
                critic_perception = (
                    tuple(torch.stack([item[i] for item in height_history]).flatten(0, 1) for i in range(2))
                    if self.recurrent else torch.stack(latents).flatten(0, 1) if latents else None
                )
                critic_context = {}
                if self.cfg.recovery_enabled:
                    state_ages = seed["ep_step"][None] + torch.arange(horizon + 1, device=self.device)[:, None]
                    remaining = remaining_protection(seed["recovery_role"][None], state_ages)
                    seed_bodies = self.mdp.kin.forward(*state0)
                    critic_bodies = tuple(
                        torch.cat((start[None], future.detach().reshape(horizon, batch, *future.shape[1:])), 0).flatten(0, 1)
                        for start, future in zip(seed_bodies, bodies, strict=True)
                    )
                    critic_context = {"remaining": remaining.flatten(), "body_state": critic_bodies}
                critic_x = self._critic_input(
                    flat_states,
                    critic_reference,
                    xi_with_seed.flatten(0, 1),
                    goals_with_seed.flatten(0, 1),
                    critic_perception,
                    **critic_context,
                )
                critic_x = critic_x.view(self.cfg.horizon + 1, self.cfg.policy_batch, -1)
                norm_input = critic_x.flatten(0, 1)
                if self.recurrent:
                    norm_input = self.critic.normalization_features(norm_input)
                valid_states = torch.cat((reference0.valid[None], valid_reference), 0)
                self._update_running_norms((self.critic.norm,), (norm_input[valid_states.flatten()],))
                values = self._bootstrap_value(self.critic(critic_x))
                returns = values[-1]
                targets_descending = []
                for step in range(self.cfg.horizon - 1, -1, -1):
                    returns = rewards[step].detach() + self.cfg.gamma * (1.0 - termination[step].detach()) * (
                        (1.0 - self.cfg.gae_lambda) * values[step + 1] + self.cfg.gae_lambda * returns
                    )
                    targets_descending.append(returns)
                targets = torch.stack(targets_descending[::-1])
                critic_inputs = critic_x[:-1].reshape(-1, critic_x.shape[-1])
                target_values = targets.reshape(-1)
                valid_critic = valid_reference.reshape(-1)
                critic_weight = (torch.ones_like(valid_reference, dtype=targets.dtype) * replay_weight).reshape(-1)

            count = critic_inputs.shape[0]
            minibatch = max(1, count // self.cfg.critic_minibatches)
            critic_loss = torch.zeros((), device=self.device)
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(count, device=self.device)
                for index in range(self.cfg.critic_minibatches):
                    selected = permutation[index * minibatch : (index + 1) * minibatch]
                    if selected.numel() == 0:
                        continue
                    prediction = self.critic(critic_inputs[selected])
                    # Paper Eq. (3): time SUM / batch MEAN. The reference code
                    # instead takes a time-and-batch mean (a factor H smaller).
                    squared_error = (prediction - target_values[selected]).square()
                    critic_loss = self.cfg.horizon * torch.where(
                        valid_critic[selected], squared_error * critic_weight[selected], 0.0
                    ).mean()
                    self.opt_critic.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    self._average_gradients(self.critic)
                    critic_grad = self._clip_gradient_norm("critic update", self.critic)
                    self.opt_critic.step()
                    self.optimizer_steps += 1

            actor_metrics.update(
                {
                    "Loss/critic": critic_loss.detach(),
                    "Gradient/critic": critic_grad.detach(),
                }
            )
        return actor_metrics

    def world_model_diagnostics(self, horizons: tuple[int, ...] = (1, 8, 32)) -> dict[str, torch.Tensor]:
        available = int(getattr(self.buffer, "size", self.cfg.buffer_steps))
        wanted = min(max(horizons), available)
        if wanted < min(horizons):
            return {}
        sample = self.buffer.sample_windows(self.cfg.wm_batch, wanted)
        metrics: dict[str, torch.Tensor] = {}
        if self.recurrent:
            with torch.no_grad():
                rollout = self._clean_world_model_rollout(sample, wanted)
            for horizon in horizons:
                if horizon <= wanted:
                    count = sample["mask"][:, horizon - 1].sum().clamp_min(1)
                    metrics[f"Model/physical_rmse_{horizon}"] = (rollout["physical_errors"][horizon - 1] / count).sqrt()
                    metrics[f"Model/running_std_normalized_rmse_{horizon}"] = (rollout["normalized_errors"][horizon - 1] / count).sqrt()
            metrics.update({
                f"{name}_horizon_{wanted}": value
                for name, value in rollout["scan_metrics"].items()
                if name.startswith("Perception/")
            })
            return metrics
        std = torch.maximum(self._norm_std(self.wm.out_norm), self._floor)
        with torch.no_grad():
            state = self._as_state(sample["s"][:, 0])
            latent = self._latent_at(sample, 0) if self.perception is not None else None
            for step in range(wanted):
                if self.perception is None:
                    state = self.wm.step(
                        state,
                        sample["a"][:, step],
                        sample["a_prev"][:, step],
                        self.normalize_xi(sample["xi"][:, step]),
                    )
                else:
                    state, latent = self.wm.step(
                        state,
                        sample["a"][:, step],
                        sample["a_prev"][:, step],
                        self.normalize_xi(sample["xi"][:, step]),
                        latent,
                    )
                target = self._as_state(sample["s1"][:, step])
                error = self.wm.state_error(state, target)
                normalized = (error[..., :WM_OUT] / std[..., :WM_OUT]).square()
                mask = sample["mask"][:, step]
                denominator = mask.sum().clamp_min(1)
                for horizon in horizons:
                    if step + 1 == horizon:
                        metrics[f"Model/physical_rmse_{horizon}"] = torch.sqrt(
                            (error[..., :WM_OUT].square().mean(-1) * mask).sum()
                            / denominator
                        )
                        metrics[f"Model/running_std_normalized_rmse_{horizon}"] = torch.sqrt(
                            (normalized.mean(-1) * mask).sum() / denominator
                        )
                if step + 1 in horizons:
                    groups = {
                        "root_pos": slice(0, 3),
                        "root_rot": slice(3, 9),
                        "root_linvel": slice(9, 12),
                        "root_angvel": slice(12, 15),
                        "joint_pos": slice(15, 44),
                        "joint_vel": slice(44, 73),
                    }
                    for name, group in groups.items():
                        metrics[f"Model/physical_{name}_rmse_{step + 1}"] = torch.sqrt(
                            (error[..., group].square().mean(-1) * mask).sum()
                            / denominator
                        )
                        metrics[f"Model/running_std_normalized_{name}_rmse_{step + 1}"] = torch.sqrt(
                            (normalized[..., group].mean(-1) * mask).sum()
                            / denominator
                        )
        return metrics

    def _log(self, *, final: bool = False) -> None:
        if not self._pending and not final:
            return
        local_sums = {
            name: torch.stack(values).sum()
            for name, values in self._pending.items()
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
            summed_diagnostics = {"Sampling/visits", "Sampling/failures", "Sampling/visited_bins"}
            summed_diagnostics.update(
                name for name in diagnostics
                if name.startswith("Sampling/") and name.endswith(("_global_candidates", "_supported_bins"))
            )
            diagnostics["Sampling/mastery_mean"] = diagnostics["Sampling/mastery_mean"] * self.motion.num_bins
            if self.distributed is not None and self.distributed.enabled:
                # One log-boundary collective obtains both extrema; max(x) = -min(-x).
                extrema = torch.stack((diagnostics["Sampling/mastery_min"], -diagnostics["Sampling/mastery_max"]))
                torch.distributed.all_reduce(extrema, op=torch.distributed.ReduceOp.MIN)
                diagnostics["Sampling/mastery_min"] = extrema[0]
                diagnostics["Sampling/mastery_max"] = -extrema[1]
        diagnostic_count = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=self.device,
        )
        if self.distributed is not None:
            gathered_names = self.distributed.all_gather_objects(
                tuple(local_sums)
            )
            metric_names = sorted(
                {name for names in gathered_names for name in names}
            )
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
            if sharded and name == "Sampling/mastery_mean" else diagnostic_count
            for name in diagnostics
        )
        if metric_sums:
            if self.distributed is not None:
                metric_sums, metric_counts = (
                    self.distributed.sum_and_count_tensors(
                        metric_sums,
                        metric_counts,
                    )
                )
            values = (
                torch.stack(
                    tuple(
                        value if name in summed_diagnostics else value / count.clamp_min(1.0)
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
                "Training/policy_updates_enabled": float(self.iteration > self.cfg.wm_only_iters),
                "Training/collection_action_limit": _COLLECTION_ACTION_LIMIT,
                "Training/bound_bootstrap_value": float(self.cfg.bound_bootstrap_value),
                "Training/bootstrap_upper_bound": MAX_STEP_REWARD / (1.0 - self.cfg.gamma),
            }
        )
        metrics.update(self.profiler.drain())
        # The model timer measures one minibatch; report total time per update.
        for name in ("collection", "actor", "critic", "model", "pmg_prior", "pmg_execution",
                     "pmg_prepare_per_control_step"):
            key = f"Performance/{name}"
            if key in metrics:
                batches = self.cfg.wm_batches if name == "model" else 1
                metrics[f"Performance/{name}_s"] = metrics.pop(key) * batches / 1000
        now = time.monotonic()
        metrics["Performance/iterations_per_second"] = (self.iteration - self._last_log_iteration) / max(now - self._last_log_time, 1e-9)
        self._last_log_time = now
        self._last_log_iteration = self.iteration
        if self.device.type == "cuda":
            metrics.update(self.cuda_cache.metrics())
            metrics["Memory/allocated_gib"] = torch.cuda.memory_allocated(self.device) / 2**30
            metrics["Memory/reserved_gib"] = torch.cuda.memory_reserved(self.device) / 2**30
            metrics["Memory/max_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        is_main = self.distributed is None or self.distributed.is_main
        if self.logger is not None and is_main:
            self.logger.log(metrics, self.iteration)
        if is_main:
            printable = " | ".join(
                f"{key} {value:.4g}"
                for key, value in metrics.items()
                if key.startswith(("Train/", "Model/loss", "Loss/", "Training/", "Counts/"))
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
            if self.iteration % self.cfg.log_every == 0 or self.iteration == self.cfg.iterations:
                self._log(final=True)
            saved_this_iteration = bool(self.cfg.log_dir) and (
                self.iteration % self.cfg.save_every == 0 or self.iteration == self.cfg.iterations
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
            if self.evaluation_callback is not None and self.cfg.eval_every and (
                self.iteration % self.cfg.eval_every == 0
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

    def _environment_stats(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name in ("bin_G", "bin_w_avg"):
            value = getattr(self.env, name, None)
            if torch.is_tensor(value):
                result[name] = value.detach().cpu()
        return result

    def checkpoint(
        self,
        *,
        rank_rng_states: list[dict[str, Any]] | None = None,
        sampler_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._check_physics()
        if rank_rng_states is None:
            local_rng = {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
            }
            if self.device.type == "cuda":
                local_rng["cuda"] = torch.cuda.get_rng_state(self.device).cpu()
            legacy_rng = local_rng
        else:
            legacy_rng = rank_rng_states[0]
        checkpoint = {
            "format": ("supermimic-" + self.cfg.robot_name.replace("_", "-") + "-v1"),
            "policy": self.policy.state_dict(),
            "wm": self.wm.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_policy": self.opt_policy.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            # Absent for pure-proprioception checkpoints; the loader only
            # insists on it when the saved config enabled perception.
            **(
                {"perception": self.perception.state_dict()}
                if self.perception is not None
                else {}
            ),
            "iteration": self.iteration,
            "sim_steps": self.sim_steps,
            "transitions": self.transitions,
            "windows": self.windows,
            "local_counts": {
                "sim_steps": self.sim_steps,
                "transitions": self.transitions,
                "windows": self.windows,
            },
            "global_counts": {
                "sim_steps": self.global_sim_steps,
                "transitions": self.global_transitions,
                "windows": self.global_windows,
            },
            "optimizer_steps": self.optimizer_steps,
            "step_accumulator": self.step_accumulator,
            "config": self.cfg.as_dict(),
            "provenance": self.provenance,
            "environment_stats": self._environment_stats() if self.cfg.motion_sharding == "none" else {},
            # Keep the legacy rank-0 entry for existing checkpoint tooling.
            "rng": legacy_rng,
        }
        if rank_rng_states is not None:
            checkpoint["rng_by_rank"] = rank_rng_states
        sampler = getattr(self, "sampler", None)
        if sampler is not None:
            if sampler_state is None and sampler.sharded:
                if sampler.world_size > 1:
                    raise ValueError("Sharded DDP checkpoints require save_checkpoint on every rank.")
                sampler_state = {
                    "format": "supermimic-mastery-sharded-v1", "world_size": 1,
                    "states_by_rank": [sampler.state_dict()],
                }
            checkpoint["sampler"] = sampler.state_dict() if sampler_state is None else sampler_state
        return checkpoint

    def save_checkpoint(self, path: Path) -> None:
        self._check_physics()
        # Every rank participates before rank 0 writes. Do not lose pending
        # evidence from the other ranks or count it once per rank on restore.
        sampler_state = None
        if self.sampler is not None:
            if self.sampler.sharded:
                local_state = {
                    name: value.detach().cpu() if torch.is_tensor(value) else value
                    for name, value in self.sampler.state_dict().items()
                }
                states = (self.distributed.all_gather_objects(local_state)
                          if self.distributed is not None else [local_state])
                sampler_state = {
                    "format": "supermimic-mastery-sharded-v1",
                    "world_size": self.sampler.world_size, "states_by_rank": states,
                }
            else:
                sampler_state = self.sampler.state_dict(distributed=True)
        if self.distributed is not None:
            rank_rng_states = self.distributed.gather_rank_rng_states()
        else:
            rank_rng_states = None
        if self.distributed is not None and not self.distributed.is_main:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(self.checkpoint(rank_rng_states=rank_rng_states, sampler_state=sampler_state), temporary)
        os.replace(temporary, path)

    def load_weights(self, path: str | os.PathLike[str]) -> None:
        """Initialize a fresh run without importing optimizer or old motion-catalog state.

        Running normalization belongs to the learned network mapping, so its
        buffers accompany the weights. Replay, RNG, samplers and clocks stay new.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != ("supermimic-" + self.cfg.robot_name.replace("_", "-") + "-v1"):
            raise ValueError("--init_weights requires a SuperMimic checkpoint")
        saved = checkpoint["config"]
        for name in ("hist", "goal_frames", "goal_mode", "wm_target_mode", "actor_relative_orientation",
                     "recovery_enabled", "perception", "perception_arch", "critic_perception",
                     "perception_latent_dim", "height_scan_frame", "height_scan_resolution"):
            if saved.get(name) != getattr(self.cfg, name):
                raise ValueError(f"--init_weights {name} differs from the checkpoint")
        if tuple(saved["height_scan_size"]) != tuple(self.cfg.height_scan_size):
            raise ValueError("--init_weights height_scan_size differs from the checkpoint")
        for name in ("policy", "wm", "critic", "perception"):
            module = getattr(self, name)
            if module is not None:
                module.load_state_dict(checkpoint[name])
        self.provenance["weights_initialization"] = {
            "checkpoint": str(Path(path).resolve()), "source_iteration": checkpoint["iteration"],
            "normalization": "inherited", "training_state": "fresh",
        }

    def load_checkpoint(self, path: str | os.PathLike[str], *, restore_rng: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != ("supermimic-" + self.cfg.robot_name.replace("_", "-") + "-v1"):
            raise ValueError(f"unsupported SuperMimic checkpoint format: {checkpoint.get('format')!r}")
        saved_config = checkpoint.get("config", {})
        if saved_config.get("motion_sharding", "none") != self.cfg.motion_sharding:
            raise ValueError("checkpoint motion_sharding differs; full resume requires the same partition mode")
        sampler = getattr(self, "sampler", None)
        sampler_state = checkpoint.get("sampler")
        if self.cfg.motion_sharding == "group":
            if sampler is None or not sampler.sharded:
                raise ValueError("Sharded full resume requires a rank-local mastery sampler.")
            if not isinstance(sampler_state, dict) or sampler_state.get("format") != "supermimic-mastery-sharded-v1":
                raise ValueError("Sharded checkpoint must contain sampler states for every rank.")
            if sampler_state.get("world_size") != sampler.world_size:
                raise ValueError("Sharded checkpoint world size differs; full resume requires the same partition.")
            states = sampler_state.get("states_by_rank")
            if not isinstance(states, list) or len(states) != sampler.world_size:
                raise ValueError("Sharded checkpoint is missing rank-local sampler states.")
            if any(not isinstance(state, dict) or state.get("rank") != rank
                   or state.get("world_size") != sampler.world_size
                   for rank, state in enumerate(states)):
                raise ValueError("Sharded checkpoint sampler rank/world size identities differ.")
            sampler_state = states[sampler.rank]
        saved_size = saved_config.get("height_scan_size", HEIGHT_SCAN_SIZE)
        saved_resolution = saved_config.get("height_scan_resolution", HEIGHT_SCAN_RESOLUTION)
        height_scan_shape(saved_size, saved_resolution)
        if (tuple(saved_size) != tuple(self.cfg.height_scan_size)
                or saved_resolution != self.cfg.height_scan_resolution):
            raise ValueError("checkpoint height_scan geometry differs; changing size/resolution requires a fresh run")
        if saved_config.get("height_scan_frame", "yaw") != self.cfg.height_scan_frame:
            raise ValueError("checkpoint height_scan_frame differs; scan semantics and encoder widths require a fresh run")
        if bool(saved_config.get("recovery_enabled", False)) != self.cfg.recovery_enabled:
            raise ValueError(
                "checkpoint recovery_enabled mode differs from this run; full resume requires the same "
                "critic width and optimizer state (cross-mode migration is not supported)"
            )
        if bool(saved_config.get("actor_relative_orientation", False)) != self.cfg.actor_relative_orientation:
            raise ValueError("checkpoint actor_relative_orientation differs; full resume requires matching actor inputs")
        # The perception branch changes the policy and world-model input widths,
        # so a mismatch must be reported as a perception mismatch rather than as
        # a confusing tensor-shape error during load.
        if self.perception is not None and "perception" not in checkpoint:
            raise ValueError(
                "this run was configured with perception but the checkpoint has "
                "no perception state (it is a pure-proprioception model)"
            )
        if self.perception is None and "perception" in checkpoint:
            raise ValueError(
                "the checkpoint carries a perception branch but this run was "
                "configured without config.perception"
            )
        if self.perception is not None and saved_config.get("perception_arch", "latent") != self.cfg.perception_arch:
            raise ValueError("checkpoint perception_arch does not match config.perception_arch")
        if bool(saved_config.get("critic_perception", False)) != self.cfg.critic_perception:
            raise ValueError("checkpoint critic_perception does not match config.critic_perception")
        if self.perception is not None and saved_config.get("perception_latent_dim") != self.cfg.perception_latent_dim:
            raise ValueError("checkpoint perception_latent_dim does not match config.perception_latent_dim")
        saved_critic_mean = checkpoint["critic"].get("norm.mean")
        if torch.is_tensor(saved_critic_mean) and saved_critic_mean.numel() != self.critic.norm.mean.numel():
            raise ValueError(
                f"checkpoint critic input width {saved_critic_mean.numel()} does not match "
                f"Critic({self.critic.in_dim}); critic_perception={self.cfg.critic_perception}"
            )
        if sampler is not None:
            if sampler_state is not None:
                sampler.load_state_dict(sampler_state)
            else:
                print("Checkpoint has no mastery sampler state; starting uniform 25-step bins.", flush=True)
        self.policy.load_state_dict(checkpoint["policy"])
        self.wm.load_state_dict(checkpoint["wm"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.opt_policy.load_state_dict(checkpoint["opt_policy"])
        self.opt_wm.load_state_dict(checkpoint["opt_wm"])
        self.opt_critic.load_state_dict(checkpoint["opt_critic"])
        if self.perception is not None:
            self.perception.load_state_dict(checkpoint["perception"])
        self.iteration = int(checkpoint["iteration"])
        self.cuda_cache.reset_clock()
        self.sim_steps = int(checkpoint["sim_steps"])
        self.transitions = int(checkpoint["transitions"])
        self.windows = int(checkpoint["windows"])
        global_counts = checkpoint.get("global_counts")
        if global_counts is not None:
            self.global_sim_steps = int(global_counts["sim_steps"])
            self.global_transitions = int(global_counts["transitions"])
            self.global_windows = int(global_counts["windows"])
        else:
            # Legacy checkpoints only had one learner. Their counters were global.
            self.global_sim_steps = self.sim_steps
            self.global_transitions = self.transitions
            self.global_windows = self.windows
        self.optimizer_steps = int(checkpoint["optimizer_steps"])
        self.step_accumulator = float(checkpoint["step_accumulator"])
        for name, value in checkpoint.get("environment_stats", {}).items():
            if hasattr(self.env, name):
                # Managers own these tensors. Rebinding an adapter attribute
                # would leave the actual command sampler at its initial values.
                getattr(self.env, name).copy_(value)
        if restore_rng:
            if self.distributed is not None:
                self.distributed.restore_checkpoint_rng(checkpoint)
            elif checkpoint.get("rng"):
                torch.set_rng_state(checkpoint["rng"]["torch"].cpu())
                if self.device.type == "cuda" and checkpoint["rng"].get("cuda") is not None:
                    cuda_state = checkpoint["rng"]["cuda"]
                    if isinstance(cuda_state, (list, tuple)):
                        index = (
                            self.device.index
                            if self.device.index is not None
                            else torch.cuda.current_device()
                        )
                        cuda_state = cuda_state[index]
                    torch.cuda.set_rng_state(cuda_state.cpu(), device=self.device)
        # Replay tensors are intentionally not checkpointed; train() refills them.
        return checkpoint
