"""GPU mastery evidence and bounded reset distributions for SuperMimic."""

from __future__ import annotations

import torch
import torch.distributed as dist

from mimic_baseline.tasks.supermimic.data.motion import BIN_FRAMES, MotionLib


@torch.no_grad()
def bounded_probabilities(
    weights: torch.Tensor, lower: torch.Tensor | float, upper: torch.Tensor | float
) -> torch.Tensor:
    """Project normalized nonnegative weights onto a bounded probability simplex.

    Bounds must be feasible: ``0 <= lower <= upper`` and
    ``sum(lower) <= 1 <= sum(upper)``. Scalar bounds broadcast. Zero weights
    request a uniform target. All arithmetic stays on the input device; the
    additive projection preserves bounds without a final renormalization.
    """
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("weights must be a nonempty vector.")
    if weights.dtype not in (torch.float32, torch.float64):
        raise TypeError("weights must be float32 or float64.")
    lower = torch.as_tensor(lower, device=weights.device, dtype=weights.dtype).expand_as(weights)
    upper = torch.as_tensor(upper, device=weights.device, dtype=weights.dtype).expand_as(weights)
    target = weights.double()
    total = target.sum()
    target = torch.where(
        total > 0, target / total.clamp_min(torch.finfo(torch.float64).tiny),
        torch.full_like(target, 1.0 / weights.numel()),
    )
    lo, hi = lower.double(), upper.double()
    left, right = (lo - target).amin(), (hi - target).amax()
    for _ in range(32):
        midpoint = (left + right) * 0.5
        below = torch.clamp(target + midpoint, min=lo, max=hi).sum() < 1.0
        left = torch.where(below, midpoint, left)
        right = torch.where(below, right, midpoint)
    result = torch.clamp(target + (left + right) * 0.5, min=lo, max=hi).to(weights.dtype)
    # Correct reduction/cast roundoff using available slack, never rescaling
    # coordinates that have already reached a floor or cap.
    for _ in range(2):
        residual = 1.0 - result.sum()
        slack = torch.where(residual >= 0, upper - result, result - lower)
        index = slack.argmax().reshape(1)
        correction = torch.clamp(
            residual.reshape(1),
            min=(lower - result).gather(0, index),
            max=(upper - result).gather(0, index),
        )
        result.scatter_add_(0, index, correction)
        result.clamp_(min=lower, max=upper)
    return result


class MasterySampler:
    """Score 25-transition attempts using pre-step reference frame indices.

    Call ``reset`` after environment reset, then ``record_step(fpre, ...)`` once
    per real vector step, before autoreset. Quality is positive tracking quality
    in [0, 1]. Only uninterrupted full-bin passes receive mean quality; a real
    termination receives zero even for partial entry, and wins on the final
    step. Nonfailure partial exits do not produce evidence.

    DDP uses the default process group and requires identical num_envs,
    update_interval, sharding mode and synchronous calls on every rank. Shared
    catalogs pool evidence; sharded catalogs adapt independently. The Python
    clock counts fixed-size real batches, with no per-step host tensor reads.
    An update runs at the first vector-step boundary crossing each interval;
    a batch spanning multiple intervals is aggregated in one update.

    Checkpoints deliberately exclude per-environment partial attempts because
    the environment and replay reset on resume. Call reset before recording.
    """

    def __init__(
        self, motion: MotionLib, num_envs: int, device: str | torch.device, *,
        distributed: bool = False, update_interval: int = 2**20,
    ) -> None:
        if num_envs < 1 or update_interval < 1 or motion.num_bins < 1:
            raise ValueError("num_envs, update_interval and num_bins must be positive.")
        self.motion = motion
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.distributed = bool(distributed)
        self.sharded = bool(getattr(motion, "sharded", False))
        self.update_interval = int(update_interval)
        self.rank, self.world_size = 0, 1
        if self.distributed:
            if not dist.is_initialized():
                raise RuntimeError("distributed=True requires an initialized process group.")
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
            configuration = torch.tensor(
                [num_envs, update_interval, int(self.sharded), 0 if self.sharded else motion.num_bins],
                device=self.device, dtype=torch.long,
            )
            configurations = [torch.empty_like(configuration) for _ in range(self.world_size)]
            dist.all_gather(configurations, configuration)
            if any(not torch.equal(configuration, other) for other in configurations):
                raise ValueError(
                    "All ranks require identical num_envs, update_interval and sharding mode; "
                    "shared catalogs also require identical num_bins."
                )
        if self.sharded and not getattr(motion, "shard_signature", None):
            raise ValueError("Sharded motion requires a nonempty shard_signature for its local catalog.")

        bins = motion.num_bins
        self.probabilities = torch.full((bins,), 1.0 / bins, device=self.device)
        self.mastery = torch.zeros(bins, device=self.device)
        self._wm_probabilities = 0.8 * self.probabilities + 0.2 / bins
        self.pending_sums = torch.zeros(bins, device=self.device, dtype=torch.float64)
        self.pending_counts = torch.zeros(bins, device=self.device, dtype=torch.long)
        self.pending_failures = torch.zeros_like(self.pending_counts)
        # Shared catalogs accumulate global evidence; sharded catalogs stay local.
        self.visit_counts = torch.zeros_like(self.pending_counts)
        self.fail_counts = torch.zeros_like(self.pending_counts)
        self.updates = 0
        self.global_transitions = 0
        self._updates_tensor = torch.zeros((), device=self.device, dtype=torch.long)
        self._transitions_tensor = torch.zeros_like(self._updates_tensor)
        self._min_ratio = torch.ones((), device=self.device)
        self._max_ratio = torch.ones((), device=self.device)
        self._expected_frame = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._bin = torch.zeros_like(self._expected_frame)
        self._steps = torch.zeros_like(self._expected_frame)
        self._quality_sum = torch.zeros(num_envs, device=self.device)
        self._full_entry = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._active = torch.zeros_like(self._full_entry)

    @property
    def actor_probabilities(self) -> torch.Tensor:
        return self.probabilities

    @property
    def wm_probabilities(self) -> torch.Tensor:
        """Persistent cache: identity/version are consumed by replay sampling."""
        return self._wm_probabilities

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        bins = self.motion.frame_bin[frames]
        self._expected_frame[env_ids] = frames
        self._bin[env_ids] = bins
        self._steps[env_ids] = 0
        self._quality_sum[env_ids] = 0
        self._full_entry[env_ids] = frames == self.motion.bin_start[bins]
        self._active[env_ids] = frames < self.motion.clip_end[self.motion.clip_id[frames]]

    @torch.no_grad()
    def record_step(
        self, frames: torch.Tensor, quality: torch.Tensor,
        terminated: torch.Tensor, truncated: torch.Tensor,
        *, active_mask: torch.Tensor | None = None,
    ) -> None:
        """Record one real transition per env; frames are BEFORE that step."""
        bins = self.motion.frame_bin[frames]
        starts = self.motion.bin_start[bins]
        ends = self.motion.clip_end[self.motion.clip_id[frames]]
        valid = self._active & (frames < ends)
        if active_mask is not None:
            valid = valid & active_mask
        consecutive = valid & (frames == self._expected_frame)
        continuing = consecutive & (bins == self._bin)
        steps = torch.where(continuing, self._steps, 0) + 1
        quality_sum = torch.where(continuing, self._quality_sum, 0.0) + quality.clamp(0.0, 1.0)
        full_entry = torch.where(continuing, self._full_entry, consecutive & (frames == starts))
        failure = valid & terminated
        complete = (
            consecutive & full_entry & (steps == BIN_FRAMES)
            & (frames == starts + BIN_FRAMES - 1) & ~terminated
        )
        self.pending_sums.index_add_(
            0, bins, torch.where(complete, quality_sum / BIN_FRAMES, 0.0).double()
        )
        self.pending_counts.index_add_(0, bins, (complete | failure).long())
        self.pending_failures.index_add_(0, bins, failure.long())
        self._expected_frame.copy_(frames + 1)
        self._bin.copy_(bins)
        self._steps.copy_(steps)
        self._quality_sum.copy_(quality_sum)
        self._full_entry.copy_(full_entry)
        self._active.copy_(valid & ~terminated & ~truncated & (frames + 1 < ends))

        old_interval = self.global_transitions // self.update_interval
        transitions = self.num_envs * self.world_size
        self.global_transitions += transitions
        self._transitions_tensor.add_(transitions)
        if self.global_transitions // self.update_interval > old_interval:
            self._update()

    def _pending(self, aggregate: bool) -> torch.Tensor:
        packed = torch.stack((self.pending_sums, self.pending_counts.double(), self.pending_failures.double()))
        if aggregate and self.distributed and not self.sharded:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return packed

    def _set_probabilities(self, probabilities: torch.Tensor) -> None:
        # This comparison is only at infrequent update/load boundaries. Keeping
        # versions stable between changes avoids rebuilding replay's bin cache.
        if not torch.equal(probabilities, self.probabilities):
            self.probabilities.copy_(probabilities)
            self._wm_probabilities.copy_(0.8 * self.probabilities + 0.2 / self.motion.num_bins)

    def _update(self) -> None:
        sums, counts, failures = self._pending(aggregate=True).unbind(0)
        alpha = -torch.expm1(-counts / 50.0)
        means = sums / counts.clamp_min(1.0)
        self.mastery.copy_(torch.where(counts > 0, self.mastery + alpha * (means - self.mastery), self.mastery))
        self.visit_counts.add_(counts.long())
        self.fail_counts.add_(failures.long())
        self.pending_sums.zero_()
        self.pending_counts.zero_()
        self.pending_failures.zero_()

        bins = self.motion.num_bins
        difficulty = (1.0 - self.mastery).square()
        total = difficulty.sum()
        normalized = torch.where(total > 0, difficulty / total.clamp_min(1e-30), 1.0 / bins)
        target = 0.05 / bins + 0.95 * normalized
        lower = self.probabilities.div(1.2).clamp_min(0.05 / bins)
        upper = self.probabilities.mul(1.2).clamp_max(min(1.0, 1.0 / bins + 0.05))
        probabilities = bounded_probabilities(target, lower, upper)
        ratios = probabilities / self.probabilities
        self._min_ratio.copy_(ratios.amin())
        self._max_ratio.copy_(ratios.amax())
        self._set_probabilities(probabilities)
        self.updates += 1
        self._updates_tensor.add_(1)

    @torch.no_grad()
    def sample_frames(
        self, count: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if count < 0:
            raise ValueError("count must be nonnegative.")
        if count == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty.clone()
        bins = torch.multinomial(self.probabilities, count, replacement=True, generator=generator)
        offset = torch.randint(BIN_FRAMES, (count,), device=self.device, generator=generator)
        at_start = torch.rand(count, device=self.device, generator=generator) < 0.5
        return self.motion.bin_start[bins] + torch.where(at_start, 0, offset), bins

    @torch.no_grad()
    def state_dict(self, *, distributed: bool = False) -> dict:
        """Snapshot completed evidence and clock, never partial env attempts.

        For shared catalogs, ``distributed=True`` pools pending copies on every
        rank without changing live accumulators. Sharded snapshots always stay
        rank-local; the trainer saves all ranks at the checkpoint boundary.
        """
        sums, counts, failures = self._pending(aggregate=distributed).unbind(0)
        return {
            "format": "supermimic-mastery-v1",
            "scope": "global" if distributed and not self.sharded else "local",
            "rank": self.rank,
            "world_size": self.world_size,
            **({"sharded": True, "shard_signature": self.motion.shard_signature} if self.sharded else {}),
            "update_interval": self.update_interval,
            "updates": self.updates,
            "global_transitions": self.global_transitions,
            "probabilities": self.probabilities.clone(),
            "mastery": self.mastery.clone(),
            "pending_sums": sums.clone(),
            "pending_counts": counts.long(),
            "pending_failures": failures.long(),
            "visit_counts": self.visit_counts.clone(),
            "fail_counts": self.fail_counts.clone(),
            "min_ratio": self._min_ratio.clone(),
            "max_ratio": self._max_ratio.clone(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        """Restore local evidence, or shared-catalog pooled evidence on any topology.

        Pooled pending evidence is owned by rank zero after restore; other ranks
        start with zero pending. This preserves integral counts and ensures the
        next collective counts saved evidence exactly once. Global cumulative
        counts, mastery and probabilities are replicated, never summed again.
        """
        if state.get("format") != "supermimic-mastery-v1":
            raise ValueError("Unsupported mastery sampler checkpoint format.")
        scope = state.get("scope")
        if scope not in {"global", "local"}:
            raise ValueError("Unknown mastery sampler checkpoint scope.")
        if bool(state.get("sharded", False)) != self.sharded:
            raise ValueError("Checkpoint sampler sharding mode differs; full resume requires matching catalogs.")
        if self.sharded:
            if scope != "local":
                raise ValueError("Sharded sampler state must be rank-local.")
            if state.get("shard_signature") != self.motion.shard_signature:
                raise ValueError("Checkpoint shard_signature differs from the local motion catalog.")
        if scope == "local" and (state["rank"] != self.rank or state["world_size"] != self.world_size):
            raise ValueError("Rank-local sampler state requires the same rank and world size.")
        names = ("mastery", "pending_sums", "pending_counts", "pending_failures", "visit_counts", "fail_counts")
        for name in ("probabilities", *names):
            if state[name].shape != self.probabilities.shape:
                raise ValueError(f"Checkpoint {name} does not match the motion bin catalog.")
        self.update_interval = int(state["update_interval"])
        self.updates = int(state["updates"])
        self.global_transitions = int(state["global_transitions"])
        if self.update_interval < 1 or self.updates < 0 or self.global_transitions < 0:
            raise ValueError("Invalid mastery sampler checkpoint counters.")
        for name in names:
            destination = getattr(self, name)
            if scope == "global" and self.rank != 0 and name.startswith("pending_"):
                destination.zero_()
            else:
                destination.copy_(state[name])
        self._set_probabilities(state["probabilities"].to(self.device))
        self._min_ratio.copy_(state["min_ratio"])
        self._max_ratio.copy_(state["max_ratio"])
        self._updates_tensor.fill_(self.updates)
        self._transitions_tensor.fill_(self.global_transitions)
        self._active.zero_()
        self._full_entry.zero_()
        self._steps.zero_()
        self._quality_sum.zero_()

    def metrics(self) -> dict[str, torch.Tensor]:
        """GPU scalars; visits/failures are local for shards, global otherwise."""
        return {
            "Sampling/mastery_mean": self.mastery.mean(),
            "Sampling/mastery_min": self.mastery.amin(),
            "Sampling/mastery_max": self.mastery.amax(),
            "Sampling/probability_min": self.probabilities.amin(),
            "Sampling/probability_max": self.probabilities.amax(),
            "Sampling/entropy": -(self.probabilities * self.probabilities.log()).sum(),
            "Sampling/visits": self.visit_counts.sum(),
            "Sampling/failures": self.fail_counts.sum(),
            "Sampling/visited_bins": (self.visit_counts > 0).sum(),
            "Sampling/pending_visits_local": self.pending_counts.sum(),
            "Sampling/pending_failures_local": self.pending_failures.sum(),
            "Sampling/updates": self._updates_tensor.clone(),
            "Sampling/transitions": self._transitions_tensor.clone(),
            "Sampling/min_ratio": self._min_ratio.clone(),
            "Sampling/max_ratio": self._max_ratio.clone(),
        }


__all__ = ["MasterySampler", "bounded_probabilities"]
