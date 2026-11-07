from __future__ import annotations


import os


import random


from pathlib import Path


from typing import Any


import torch


from torch.utils.checkpoint import checkpoint


from mimic_baseline.tasks.supermimic.learning.perception import (
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
    height_scan_shape,
)


class CheckpointMixin:
    """Original Trainer methods; state is owned by Trainer."""

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
            "environment_stats": self._environment_stats()
            if self.cfg.motion_sharding == "none"
            else {},
            # Keep the legacy rank-0 entry for existing checkpoint tooling.
            "rng": legacy_rng,
        }
        if rank_rng_states is not None:
            checkpoint["rng_by_rank"] = rank_rng_states
        sampler = getattr(self, "sampler", None)
        if sampler is not None:
            if sampler_state is None and sampler.sharded:
                if sampler.world_size > 1:
                    raise ValueError(
                        "Sharded DDP checkpoints require save_checkpoint on every rank."
                    )
                sampler_state = {
                    "format": "supermimic-mastery-sharded-v1",
                    "world_size": 1,
                    "states_by_rank": [sampler.state_dict()],
                }
            checkpoint["sampler"] = (
                sampler.state_dict() if sampler_state is None else sampler_state
            )
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
                states = (
                    self.distributed.all_gather_objects(local_state)
                    if self.distributed is not None
                    else [local_state]
                )
                sampler_state = {
                    "format": "supermimic-mastery-sharded-v1",
                    "world_size": self.sampler.world_size,
                    "states_by_rank": states,
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
        torch.save(
            self.checkpoint(
                rank_rng_states=rank_rng_states, sampler_state=sampler_state
            ),
            temporary,
        )
        os.replace(temporary, path)

    def load_weights(self, path: str | os.PathLike[str]) -> None:
        """Initialize a fresh run without importing optimizer or old motion-catalog state.

        Running normalization belongs to the learned network mapping, so its
        buffers accompany the weights. Replay, RNG, samplers and clocks stay new.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != (
            "supermimic-" + self.cfg.robot_name.replace("_", "-") + "-v1"
        ):
            raise ValueError("--init_weights requires a SuperMimic checkpoint")
        saved = checkpoint["config"]
        for name in (
            "hist",
            "goal_frames",
            "goal_mode",
            "wm_target_mode",
            "actor_relative_orientation",
            "recovery_enabled",
            "perception",
            "perception_arch",
            "critic_perception",
            "perception_latent_dim",
            "height_scan_frame",
            "height_scan_resolution",
        ):
            if saved.get(name) != getattr(self.cfg, name):
                raise ValueError(f"--init_weights {name} differs from the checkpoint")
        if tuple(saved["height_scan_size"]) != tuple(self.cfg.height_scan_size):
            raise ValueError(
                "--init_weights height_scan_size differs from the checkpoint"
            )
        for name in ("policy", "wm", "critic", "perception"):
            module = getattr(self, name)
            if module is not None:
                module.load_state_dict(checkpoint[name])
        self.provenance["weights_initialization"] = {
            "checkpoint": str(Path(path).resolve()),
            "source_iteration": checkpoint["iteration"],
            "normalization": "inherited",
            "training_state": "fresh",
        }

    def load_checkpoint(
        self, path: str | os.PathLike[str], *, restore_rng: bool = True
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != (
            "supermimic-" + self.cfg.robot_name.replace("_", "-") + "-v1"
        ):
            raise ValueError(
                f"unsupported SuperMimic checkpoint format: {checkpoint.get('format')!r}"
            )
        saved_config = checkpoint.get("config", {})
        if saved_config.get("motion_sharding", "none") != self.cfg.motion_sharding:
            raise ValueError(
                "checkpoint motion_sharding differs; full resume requires the same partition mode"
            )
        sampler = getattr(self, "sampler", None)
        sampler_state = checkpoint.get("sampler")
        if self.cfg.motion_sharding == "group":
            if sampler is None or not sampler.sharded:
                raise ValueError(
                    "Sharded full resume requires a rank-local mastery sampler."
                )
            if (
                not isinstance(sampler_state, dict)
                or sampler_state.get("format") != "supermimic-mastery-sharded-v1"
            ):
                raise ValueError(
                    "Sharded checkpoint must contain sampler states for every rank."
                )
            if sampler_state.get("world_size") != sampler.world_size:
                raise ValueError(
                    "Sharded checkpoint world size differs; full resume requires the same partition."
                )
            states = sampler_state.get("states_by_rank")
            if not isinstance(states, list) or len(states) != sampler.world_size:
                raise ValueError(
                    "Sharded checkpoint is missing rank-local sampler states."
                )
            if any(
                not isinstance(state, dict)
                or state.get("rank") != rank
                or state.get("world_size") != sampler.world_size
                for rank, state in enumerate(states)
            ):
                raise ValueError(
                    "Sharded checkpoint sampler rank/world size identities differ."
                )
            sampler_state = states[sampler.rank]
        saved_size = saved_config.get("height_scan_size", HEIGHT_SCAN_SIZE)
        saved_resolution = saved_config.get(
            "height_scan_resolution", HEIGHT_SCAN_RESOLUTION
        )
        height_scan_shape(saved_size, saved_resolution)
        if (
            tuple(saved_size) != tuple(self.cfg.height_scan_size)
            or saved_resolution != self.cfg.height_scan_resolution
        ):
            raise ValueError(
                "checkpoint height_scan geometry differs; changing size/resolution requires a fresh run"
            )
        if saved_config.get("height_scan_frame", "yaw") != self.cfg.height_scan_frame:
            raise ValueError(
                "checkpoint height_scan_frame differs; scan semantics and encoder widths require a fresh run"
            )
        if (
            bool(saved_config.get("recovery_enabled", False))
            != self.cfg.recovery_enabled
        ):
            raise ValueError(
                "checkpoint recovery_enabled mode differs from this run; full resume requires the same "
                "critic width and optimizer state (cross-mode migration is not supported)"
            )
        if (
            bool(saved_config.get("actor_relative_orientation", False))
            != self.cfg.actor_relative_orientation
        ):
            raise ValueError(
                "checkpoint actor_relative_orientation differs; full resume requires matching actor inputs"
            )
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
        if (
            self.perception is not None
            and saved_config.get("perception_arch", "latent")
            != self.cfg.perception_arch
        ):
            raise ValueError(
                "checkpoint perception_arch does not match config.perception_arch"
            )
        if (
            bool(saved_config.get("critic_perception", False))
            != self.cfg.critic_perception
        ):
            raise ValueError(
                "checkpoint critic_perception does not match config.critic_perception"
            )
        if (
            self.perception is not None
            and saved_config.get("perception_latent_dim")
            != self.cfg.perception_latent_dim
        ):
            raise ValueError(
                "checkpoint perception_latent_dim does not match config.perception_latent_dim"
            )
        saved_critic_mean = checkpoint["critic"].get("norm.mean")
        if (
            torch.is_tensor(saved_critic_mean)
            and saved_critic_mean.numel() != self.critic.norm.mean.numel()
        ):
            raise ValueError(
                f"checkpoint critic input width {saved_critic_mean.numel()} does not match "
                f"Critic({self.critic.in_dim}); critic_perception={self.cfg.critic_perception}"
            )
        if sampler is not None:
            if sampler_state is not None:
                sampler.load_state_dict(sampler_state)
            else:
                print(
                    "Checkpoint has no mastery sampler state; starting uniform 25-step bins.",
                    flush=True,
                )
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
                if (
                    self.device.type == "cuda"
                    and checkpoint["rng"].get("cuda") is not None
                ):
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
