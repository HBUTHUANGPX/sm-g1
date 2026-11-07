"""Original training settings and validation, independent of simulation."""

from __future__ import annotations


import math


from dataclasses import asdict, dataclass


from typing import Any


from mimic_baseline.tasks.supermimic.learning.perception import (
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
    height_scan_shape,
)


@dataclass
class TrainingConfig:
    """Paper optimization defaults plus explicit RX-P2 environment settings."""

    self_collisions: bool = True
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
            raise ValueError(
                "config.height_raycast_backend must be warp, ovrtx, or None"
            )
        if not isinstance(self.actor_relative_orientation, bool):
            raise ValueError("config.actor_relative_orientation must be bool")
        if not math.isfinite(self.term_height) or self.term_height <= 0.0:
            raise ValueError("config.term_height must be finite and positive")
        if (
            type(self.cuda_cache_clear_interval) is not int
            or self.cuda_cache_clear_interval < 0
        ):
            raise ValueError(
                "config.cuda_cache_clear_interval must be a non-negative integer"
            )
        if self.actor_recompute not in ("none", "policy", "full"):
            raise ValueError("config.actor_recompute must be none, policy, or full")
        height_scan_shape(self.height_scan_size, self.height_scan_resolution)
        if (
            tuple(self.height_scan_size) != HEIGHT_SCAN_SIZE
            or self.height_scan_resolution != HEIGHT_SCAN_RESOLUTION
        ) and not (self.perception and self.perception_arch == "recurrent"):
            raise ValueError(
                "nondefault height_scan geometry requires recurrent perception"
            )
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
        if self.height_scan_frame == "world" and not (
            self.perception and self.perception_arch == "recurrent"
        ):
            raise ValueError("world height_scan_frame requires recurrent perception")
        for name in (
            "perception_height_weight",
            "perception_depth_weight",
            "perception_decoder_physics_ratio",
        ):
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
            raise ValueError(
                "perception loss ablations require clean recurrent perception"
            )
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
                raise ValueError(
                    "perception loss weights must be finite and non-negative"
                )
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
