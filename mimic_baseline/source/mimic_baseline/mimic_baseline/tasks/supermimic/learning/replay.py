"""Fixed-size GPU replay with episode-aware windows and start histories.

The storage layout is ``[time, env]``. In-place insertion avoids host transfers,
and sampling indexes the current logical age range so slots overwritten by the
ring are never presented as future states. The ring traversal follows the
reviewed MIT reference, with RX-P2's 27-action width and stricter overwrite
clamping.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.distributed as dist

from mimic_baseline.tasks.supermimic.core.mdp import ACT_DIM, OBS_DIM, STATE_DIM, XI_DIM
from mimic_baseline.tasks.supermimic.learning.perception import DEPTH_SHAPE, HEIGHT_SCAN_DIM


Fields = dict[str, tuple[tuple[int, ...], torch.dtype]]


class NoPMGDecisionSeeds(RuntimeError):
    """Warmup/replay currently contains no eligible generated decision seed."""


class NoMixedReplayCandidates(RuntimeError):
    """At least one rank cannot sample this mixed-replay update safely."""


FIELDS: Fields = {
    "s": ((STATE_DIM,), torch.float32),
    "s1": ((STATE_DIM,), torch.float32),
    "a": ((ACT_DIM,), torch.float32),
    "a_prev": ((ACT_DIM,), torch.float32),
    "xi": ((XI_DIM,), torch.float32),
    "f": ((), torch.long),
    "ref_id": ((), torch.long),
    "ref_phase": ((), torch.long),
    "scene_id": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
}

# Perception is stored raw, never as an encoded latent: the encoder is trained
# jointly, so a latent written at collection time means something different by
# the time the window is sampled. Depth is half precision because the encoder
# input is normalized to [0, 1] anyway; the height scan is a privileged target
# in metres and keeps enough range at the same width.
#
# Both endpoints are explicit: automatic reset can replace the next row's
# observation, even though the terminal physical transition is still valid.
# At the default 189 cells, legacy perception fields occupy 7534 B/transition.
# Recurrent replay includes an extra held capture: 11683 B with physical fields, or
# 44.57 GiB at 4096 envs x 1000 steps. Budget model/physics memory separately;
# reduce buffer_steps or num_envs rather than moving replay to CPU.
# Configurable recurrent storage is 10549 + 6 * scan_dim B, plus 1 B for recovery.
PERCEPTION_FIELDS: Fields = {
    "depth": (DEPTH_SHAPE, torch.float16),
    "depth1": (DEPTH_SHAPE, torch.float16),
    "scan": ((HEIGHT_SCAN_DIM,), torch.float16),
    "scan1": ((HEIGHT_SCAN_DIM,), torch.float16),
    "hit": ((HEIGHT_SCAN_DIM,), torch.bool),
    "hit1": ((HEIGHT_SCAN_DIM,), torch.bool),
}

# Current clean endpoints are distinct from the actor's delayed images. Store
# one held noisy capture per transition; reconstruct its short queue by indexing
# history rather than duplicating the entire queue in every replay entry.
RECURRENT_FIELDS: Fields = {
    "clean_depth": (DEPTH_SHAPE, torch.float16),
    "clean_depth1": (DEPTH_SHAPE, torch.float16),
    "camera_frame": (DEPTH_SHAPE, torch.float16),
    "camera_lag": ((), torch.int32),
}


class ReplayBuffer:
    """One-device ring buffer whose time windows have explicit semantics."""

    def __init__(
        self,
        steps: int,
        num_envs: int,
        device: str | torch.device,
        *,
        perception: bool = False,
        recurrent: bool = False,
        burn_in: int = 16,
        camera_capacity: int = 6,
        distributed: bool = False,
        recovery_enabled: bool = False,
        scan_dim: int = HEIGHT_SCAN_DIM,
        reference: object | None = None,
        pmg_fraction: float = 0.0,
        action_dim: int = ACT_DIM,
    ):
        if steps <= 0 or num_envs <= 0:
            raise ValueError(f"steps and num_envs must be positive, got {steps}, {num_envs}")
        self.T = int(steps)
        self.N = int(num_envs)
        self.device = torch.device(device)
        self.perception = bool(perception)
        self.recurrent = bool(recurrent)
        self.burn_in = int(burn_in)
        self.camera_capacity = int(camera_capacity)
        self.distributed = bool(distributed)
        self.recovery_enabled = bool(recovery_enabled)
        if isinstance(scan_dim, bool) or not isinstance(scan_dim, int) or scan_dim <= 0:
            raise ValueError("scan_dim must be a positive integer")
        self.scan_dim = scan_dim
        if self.recurrent and (not self.perception or self.burn_in < 1 or self.camera_capacity < 1):
            raise ValueError("recurrent replay needs perception and positive history lengths")
        self.ptr = 0
        self.size = 0
        self._schema_checked = False
        self._generation = 0
        # One entry per consumer keeps repeated WM updates cheap and bounds
        # memory when the caller replaces its probability tensor each iteration.
        self._sampling_cache: dict[
            str, tuple[tuple, torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]
        ] = {}
        self.sampling_metrics: dict[str, torch.Tensor] = {}

        # This is the only replay allocation; insertion and sampling reuse it.
        schema = dict(FIELDS)
        if action_dim not in (27, 29):
            raise ValueError("Supported action dimensions are RX-P2 27 and G1 29")
        self.action_dim = action_dim
        for name in ("a", "a_prev"):
            schema[name] = ((action_dim,), torch.float32)
        if self.recovery_enabled:
            schema["recovery_role"] = ((), torch.bool)
        self._perception_fields = tuple(
            name for name in PERCEPTION_FIELDS
            if not self.recurrent or name not in ("depth", "depth1")
        )
        if self.perception:
            schema.update({name: PERCEPTION_FIELDS[name] for name in self._perception_fields})
            for name in ("scan", "scan1", "hit", "hit1"):
                schema[name] = ((self.scan_dim,), PERCEPTION_FIELDS[name][1])
        if self.recurrent:
            schema.update(RECURRENT_FIELDS)
        self.data: dict[str, torch.Tensor] = {
            name: torch.empty((self.T, self.N, *tail), dtype=dtype, device=self.device)
            for name, (tail, dtype) in schema.items()
        }
        self._fields = tuple(schema)
        self.reference = None
        self.pmg_fraction = 0.0
        self._pmg_goal_frames = 10
        self._decision_cache = None
        if reference is not None:
            self.attach_pmg(reference, pmg_fraction)
        elif pmg_fraction != 0.0:
            raise ValueError("pmg_fraction requires a PMG reference")

    def attach_pmg(self, reference, actor_fraction, goal_frames=10, capture_steps=5) -> None:
        """Attach ownership/routing before collection; static-off storage is unchanged."""
        if self.size:
            raise RuntimeError("attach_pmg requires an empty buffer")
        if reference is None or not 0.0 <= actor_fraction <= 1.0 or goal_frames < 0:
            raise ValueError("PMG requires a reference, actor_fraction in [0,1], and nonnegative goal_frames")
        self.reference = reference
        self.pmg_fraction = float(actor_fraction)
        self._pmg_goal_frames = int(goal_frames)
        self._pmg_capture_steps = int(capture_steps)
        if "generated_source" not in self.data:
            self.data["generated_source"] = torch.empty((self.T, self.N), device=self.device, dtype=torch.bool)
            self.data["policy_obs"] = torch.empty((self.T, self.N, 64 + self.action_dim), device=self.device, dtype=torch.float32)
        for name, tail, dtype in (("decision_ref_id", (), torch.long), ("decision_ref_phase", (), torch.long),
                                  ("pmg_decision", (), torch.bool), ("task_command", (2,), torch.float32)):
            if name not in self.data:
                self.data[name] = torch.empty((self.T, self.N, *tail), device=self.device, dtype=dtype)
        if self.perception and "depth" not in self.data:
            self.data["depth"] = torch.empty((self.T, self.N, *DEPTH_SHAPE), device=self.device, dtype=torch.float16)
        self._fields = tuple(self.data)
        self._schema_checked = False
        self._sampling_cache.clear()
        self._decision_cache = None

    @torch.no_grad()
    def clear(self) -> None:
        """Release replay owners explicitly; environment resets never call this."""
        if self.reference is not None and self.size:
            rows = self._rows_for_ages(torch.arange(self.size, device=self.device))
            self.reference.release(self.data["ref_id"][rows])
            self.reference.release(self.data["decision_ref_id"][rows][self.data["pmg_decision"][rows]])
        self.ptr = self.size = 0
        self._schema_checked = False
        self._generation += 1
        self._sampling_cache.clear()
        self.sampling_metrics.clear()
        self._decision_cache = None

    def close(self) -> None:
        self.clear()

    @property
    def steps(self) -> int:
        return self.T

    @property
    def num_envs(self) -> int:
        return self.N

    def __len__(self) -> int:
        return self.size

    def _check_transition(self, out: Mapping[str, torch.Tensor]) -> None:
        for name in self._fields:
            tail = self.data[name].shape[2:]
            if name not in out:
                raise KeyError(f"replay transition is missing '{name}'")
            value = out[name]
            expected = (self.N, *tail)
            if not torch.is_tensor(value) or tuple(value.shape) != expected:
                actual = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise ValueError(f"replay field '{name}' must have shape {expected}, got {actual}")
            if value.device != self.device:
                raise RuntimeError(
                    f"replay field '{name}' is on {value.device}, expected {self.device}"
                )

    @torch.no_grad()
    def add(self, out: Mapping[str, torch.Tensor]) -> None:
        """Insert one vectorized transition for every environment."""
        if not self._schema_checked:
            self._check_transition(out)
            self._schema_checked = True
        if self.reference is not None:
            self.reference.retain(out["ref_id"])
            self.reference.retain(out["decision_ref_id"][out["pmg_decision"]])
            if self.size == self.T:
                self.reference.release(self.data["ref_id"][self.ptr])
                self.reference.release(self.data["decision_ref_id"][self.ptr][self.data["pmg_decision"][self.ptr]])
        for name in self._fields:
            self.data[name][self.ptr].copy_(out[name], non_blocking=True)
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        self._generation += 1
        self._sampling_cache.clear()
        self.sampling_metrics.clear()
        self._decision_cache = None

    def _decision_candidates(self) -> torch.Tensor:
        if self.reference is None:
            raise RuntimeError("decision seeds require attach_pmg")
        if self._decision_cache is None:
            ages = torch.arange(self.size, device=self.device)
            rows = self._rows_for_ages(ages) if self.size else ages
            steps = self.data["ep_step"][rows]
            eligible = self.data["pmg_decision"][rows]
            if self.recurrent:
                required = max(self.burn_in, self.camera_capacity - 1)
                eligible &= (steps <= ages[:, None]) | (ages[:, None] >= required)
                capture_step = ((steps.long() - self.data["camera_lag"][rows].long()).clamp_min(0)
                                // self._pmg_capture_steps) * self._pmg_capture_steps
                eligible &= steps - capture_step <= ages[:, None]
            self._decision_cache = eligible.flatten().nonzero(as_tuple=False).flatten()
        return self._decision_cache

    def pmg_decision_count(self) -> int:
        """Cached candidate cardinality; no scalar GPU reduction or invented seed."""
        return 0 if self.reference is None else self._decision_candidates().numel()

    def _rows_for_ages(self, ages: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("replay is empty; collect warmup transitions before sampling")
        # Callers construct ages from metadata-known bounds. Do not inspect age
        # values here: a tensor reduction would force a GPU/CPU synchronization.
        return (self.ptr - self.size + ages) % self.T

    def _gather(
        self, fields: tuple[str, ...], rows: torch.Tensor, envs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {name: self.data[name][rows, envs] for name in fields}

    def _burnin(self, age: torch.Tensor, env: torch.Tensor, ep_step: torch.Tensor) -> dict[str, torch.Tensor]:
        back = torch.arange(self.burn_in, 0, -1, device=self.device)
        available = torch.minimum(age, ep_step.long())
        valid = back[None, :] <= available[:, None]
        ages = age[:, None] - torch.minimum(back[None, :], available[:, None])
        rows = self._rows_for_ages(ages)
        fields = ("s", "a", "a_prev", "xi", "f", "ep_step", "clean_depth", "scan", "hit")
        result = self._gather(fields, rows, env[:, None].expand_as(rows))
        result["mask"] = valid
        return result

    def sampling_scale(self, kind: str) -> torch.Tensor:
        """Legacy rank target-mass metadata for the most recent weighted draw.

        Losses must use the returned per-sample sampling_weight instead: the
        locally capped proposal need not be proportional to the global target.
        Insertion clears these values; call after sampling, before the next add.
        """
        consumer = {"wm": "wm", "actor": "actor", "windows": "wm", "starts": "actor"}[kind]
        return self.sampling_metrics[f"Sampling/{consumer}_loss_scale"]

    def _weighted_logical(
        self,
        B: int,
        kind: str,
        window: int,
        motion: object,
        probabilities: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw logical row/env indices from a cached, density-corrected CDF.

        Callers pass validated actor p or WM 0.8*p + 0.2/num_bins. Unsupported
        mass is measured before renormalization, globally under DDP. A second
        bin-space projection caps each rank's actual proposal. Cached bin
        importance weights recover the global target after DDP averaging.
        """
        if not torch.is_tensor(probabilities) or probabilities.ndim != 1 or probabilities.numel() == 0:
            raise ValueError("bin_probabilities must be a nonempty rank-one tensor")
        if probabilities.device != self.device:
            raise RuntimeError("bin_probabilities must be on the replay device")
        frame_bin = getattr(motion, "frame_bin", None)
        if not torch.is_tensor(frame_bin) or frame_bin.ndim != 1 or frame_bin.dtype != torch.long:
            raise TypeError("motion must expose a rank-one long tensor frame_bin")
        if frame_bin.device != self.device:
            raise RuntimeError("motion.frame_bin must be on the replay device")
        if probabilities.numel() != motion.num_bins:
            raise ValueError("bin_probabilities must have motion.num_bins entries")
        references = (probabilities, frame_bin, motion.clip_id, motion.clip_end)
        # Inference tensors have no mutation counter: never reuse their CDF.
        versions = tuple(None if value.is_inference() else value._version for value in references)
        key = (self._generation, window, self.recurrent, self.burn_in, self.camera_capacity, self.distributed,
               tuple(id(value) for value in references), versions)
        cached = self._sampling_cache.get(kind)
        if cached is not None and cached[0] == key and None not in versions:
            cdf = cached[1]
            importance_per_bin = cached[3]
        else:
            self._sampling_cache.pop(kind, None)
            prefix = f"Sampling/{'wm' if kind == 'windows' else 'actor'}"
            q = probabilities.to(torch.float64)
            q = q / q.sum()
            ages = torch.arange(self.size - window + 1, device=self.device)
            rows = self._rows_for_ages(ages)
            frames = self.data["f"][rows]
            eligible = frames < self._motion_clip_end(motion, frames)
            if self.recurrent:
                episode_steps = self.data["ep_step"][rows]
                required = self.burn_in if kind == "windows" else max(self.burn_in, self.camera_capacity - 1)
                eligible &= (episode_steps <= ages[:, None]) | (ages[:, None] >= required)
            bins = frame_bin[frames].flatten()
            valid = eligible.flatten().to(torch.float64)
            local_counts = torch.zeros_like(q).scatter_add_(0, bins, valid)
            local_count = local_counts.sum()
            counts = local_counts
            world_size = 1
            if self.distributed:
                world_size = dist.get_world_size()
                counts = local_counts.clone()
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            supported = counts > 0
            target = q * supported
            supported_mass = target.sum()
            count = counts.sum()
            self.sampling_metrics.update({
                f"{prefix}_unsupported_mass": (q * ~supported).sum(),
                f"{prefix}_eligible_candidates": local_count,
                f"{prefix}_global_candidates": count,
                f"{prefix}_supported_bins": supported.sum(),
            })
            torch._assert_async((count > 0) & (supported_mass > 0),
                                f"no eligible {kind} candidates with supported bin probability mass")

            from mimic_baseline.tasks.supermimic.data.sampling import bounded_probabilities

            # Solve once in bin space, not once per replay row or minibatch.
            # upper_i / n_i = 8/N bounds each eligible row after redistribution.
            target = target / supported_mass
            upper = 8 * counts / count
            projected = bounded_probabilities(target, torch.zeros_like(q), upper)
            # A feasible target needs no redistribution. Preserve its tiny bin
            # masses exactly instead of injecting projection solver roundoff.
            mass = torch.where((target <= upper).all(), target, projected)
            target_per_candidate = mass / counts.clamp_min(1)
            local_target = target_per_candidate * local_counts
            local_mass = local_target.sum()
            torch._assert_async(local_mass > 0, f"no local eligible {kind} candidates with positive mass")
            if world_size > 1:
                local_target = local_target / local_mass
                local_upper = 8 * local_counts / local_count
                local_projected = bounded_probabilities(local_target, torch.zeros_like(q), local_upper)
                proposal = torch.where((local_target <= local_upper).all(), local_target, local_projected)
                proposal_per_candidate = proposal / local_counts.clamp_min(1)
                importance_per_bin = torch.where(
                    local_counts > 0,
                    world_size * target_per_candidate / proposal_per_candidate.clamp_min(torch.finfo(q.dtype).tiny),
                    0.0,
                )
            else:
                # The global projection already caps the single-rank proposal.
                # Skip a redundant projection and return exactly unit weights.
                proposal_per_candidate = target_per_candidate
                importance_per_bin = torch.ones_like(q)
            weights = proposal_per_candidate[bins] * valid
            cdf = weights.cumsum(0, dtype=torch.float64)
            self.sampling_metrics.update({
                f"{prefix}_max_candidate_probability": proposal_per_candidate.max(),
                f"{prefix}_max_target_candidate_probability": target_per_candidate.max(),
                f"{prefix}_local_mass": local_mass,
                f"{prefix}_loss_scale": world_size * local_mass,
            })
            # Retain inputs so Python cannot reuse their ids while cached.
            self._sampling_cache[kind] = (key, cdf, references, importance_per_bin)

        uniform = torch.rand(B, device=self.device, dtype=cdf.dtype, generator=generator)
        # right=True skips zero-probability plateaus, including a leading one.
        draws = torch.minimum(uniform * cdf[-1], torch.nextafter(cdf[-1], cdf.new_zeros(())))
        return torch.searchsorted(cdf, draws, right=True), importance_per_bin

    def _require_mixed_support(self, ready, message):
        ready = torch.as_tensor(ready, device=self.device, dtype=torch.long).reshape(1)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        if not ready.nonzero(as_tuple=False).numel():
            raise NoMixedReplayCandidates(message)

    def _mixed_logical(self, B, kind, window, horizon, motion, probabilities, generator, *, decision_only=False,
                       decision_weight=None):
        if decision_only:
            candidates = self._decision_candidates()
            if not candidates.numel():
                raise NoPMGDecisionSeeds("no generated decision seeds; collect beyond PMG warmup")
            draw = torch.randint(candidates.numel(), (B,), device=self.device, generator=generator)
            weight = torch.ones(B, device=self.device)
            # Decision counts are independent of sharded static bin layouts.
            if decision_weight is not None:
                weight = torch.as_tensor(decision_weight, device=self.device).expand(B)
            elif dist.is_initialized() and dist.get_world_size() > 1:
                local_count = torch.tensor(candidates.numel(), device=self.device, dtype=torch.long)
                global_count = local_count.clone()
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
                weight = (dist.get_world_size() * local_count.double() / global_count).expand(B)
            return candidates[draw], weight

        if probabilities is not None:
            if probabilities.ndim != 1 or probabilities.numel() == 0 or probabilities.numel() != motion.num_bins:
                raise ValueError("bin_probabilities must have motion.num_bins entries")
            if probabilities.device != self.device:
                raise RuntimeError("bin_probabilities must be on the replay device")
        refs = tuple(value for value in (probabilities, getattr(motion, "frame_bin", None),
                                         getattr(motion, "clip_id", None), getattr(motion, "clip_end", None))
                     if isinstance(value, torch.Tensor))
        versions = tuple(None if value.is_inference() else value._version for value in refs)
        key = (self._generation, window, horizon, self.pmg_fraction, self._pmg_goal_frames,
               tuple(id(value) for value in refs), versions)
        cache_name = "pmg:" + kind
        cached = self._sampling_cache.get(cache_name)
        if cached is not None and cached[0] == key and None not in versions:
            cdf, importance = cached[1], cached[3]
        else:
            ages = torch.arange(self.size - window + 1, device=self.device)
            rows = self._rows_for_ages(ages)
            frames = self.data["f"][rows].flatten()
            generated = self.data["ref_id"][rows].flatten() < 0
            eligible = torch.ones_like(generated)
            if self.recurrent:
                steps = self.data["ep_step"][rows]
                required = self.burn_in if kind == "windows" else max(self.burn_in, self.camera_capacity - 1)
                eligible &= ((steps <= ages[:, None]) | (ages[:, None] >= required)).flatten()
            if kind == "starts":
                phases = self.data["ref_phase"][rows].flatten()
                eligible &= ~generated | ((phases >= 0) & (phases + horizon + self._pmg_goal_frames <= 62))

            bins = probabilities.numel() if probabilities is not None else 1
            groups = torch.full_like(frames, bins)
            static = ((~generated) & eligible).nonzero(as_tuple=False).flatten()
            if static.numel():
                static_frames = frames[static]
                if probabilities is not None or kind == "starts":
                    ends = self._motion_clip_end(motion, static_frames)
                    in_clip = static_frames < ends if probabilities is not None else static_frames + horizon <= ends
                    if kind == "windows":
                        in_clip |= self.data["generated_source"][rows].flatten()[static]
                    eligible[static] &= in_clip
                groups[static] = motion.frame_bin[static_frames] if probabilities is not None else 0
            local = torch.zeros(bins + 1, device=self.device, dtype=torch.float64)
            local.scatter_add_(0, groups, eligible.double())
            global_counts = local.clone()
            world_size = 1
            sharded_sources = not self.distributed and dist.is_initialized() and dist.get_world_size() > 1
            if self.distributed:
                world_size = dist.get_world_size()
                dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
            elif sharded_sources:
                world_size = dist.get_world_size()
            ns, ng = global_counts[:-1].sum(), global_counts[-1]
            ls, lg = local[:-1].sum(), local[-1]
            tiny = torch.finfo(local.dtype).tiny

            from mimic_baseline.tasks.supermimic.data.sampling import bounded_probabilities

            if probabilities is None:
                supported_mass = local.new_ones(())
                target_static = global_counts[:-1] / ns.clamp_min(1)
            else:
                q = probabilities.double()
                q = q / q.sum()
                supported = global_counts[:-1] > 0
                target_static = q * supported
                supported_mass = target_static.sum()
                target_static = target_static / supported_mass.clamp_min(tiny)
                upper = torch.where(ns > 0, 8 * global_counts[:-1] / ns.clamp_min(1), torch.ones_like(q))
                projected = bounded_probabilities(target_static, torch.zeros_like(q), upper)
                target_static = torch.where((target_static <= upper).all(), target_static, projected)
            target_per_static = target_static / global_counts[:-1].clamp_min(1)
            local_static_mass = (target_per_static * local[:-1]).sum()
            static_rank_count = None
            if sharded_sources:
                # Static bin IDs are rank-local; only source cardinalities align.
                source_counts = torch.stack((ls, lg, ((ls > 0) & (supported_mass > 0)).double()))
                dist.all_reduce(source_counts, op=dist.ReduceOp.SUM)
                ns, ng, static_rank_count = source_counts.unbind()
            if world_size == 1:
                proposal_per_static = target_per_static
            else:
                conditional = torch.where(local_static_mass > 0,
                                          target_per_static * local[:-1] / local_static_mass.clamp_min(tiny),
                                          local[:-1] / ls.clamp_min(1))
                upper = torch.where(ls > 0, 8 * local[:-1] / ls.clamp_min(1), torch.ones_like(conditional))
                projected = bounded_probabilities(conditional, torch.zeros_like(conditional), upper)
                conditional = torch.where((conditional <= upper).all(), conditional, projected)
                proposal_per_static = conditional / local[:-1].clamp_min(1)

            occupancy = ng / (ns + ng).clamp_min(1)
            target_g = occupancy if kind == "windows" else local.new_tensor(self.pmg_fraction)
            static_support = static_rank_count > 0 if sharded_sources else (ns > 0) & (supported_mass > 0)
            target_g = torch.where(static_support, target_g, torch.ones_like(target_g))
            target_g = torch.where(ng > 0, target_g, torch.zeros_like(target_g))
            proposal_g = torch.where(ls > 0, target_g, torch.ones_like(target_g))
            proposal_g = torch.where(lg > 0, proposal_g, torch.zeros_like(proposal_g))
            proposal_s = (1 - proposal_g) * proposal_per_static
            proposal_per_group = torch.cat((proposal_s, (proposal_g / lg.clamp_min(1)).reshape(1)))
            importance_s = torch.where(proposal_s > 0,
                                       world_size * (1 - target_g) * target_per_static / proposal_s.clamp_min(tiny), 0.)
            if sharded_sources:
                importance_s = importance_s / static_rank_count.clamp_min(1)
            importance_g = torch.where(proposal_g > 0,
                                       world_size * target_g * lg / (proposal_g * ng).clamp_min(tiny), 0.)
            importance_group = torch.cat((importance_s, importance_g.reshape(1)))
            cdf = (proposal_per_group[groups] * eligible).cumsum(0, dtype=torch.float64)
            importance = importance_group[groups]
            prefix = f"Sampling/{'wm' if kind == 'windows' else 'actor'}"
            self.sampling_metrics.update({
                f"{prefix}_eligible_candidates": local.sum(),
                # Training._log sums this metric for sharded catalogs.
                f"{prefix}_global_candidates": local.sum() if sharded_sources else global_counts.sum(),
                f"{prefix}_unsupported_mass": 1 - supported_mass,
                f"{prefix}_pmg_fraction": proposal_g,
                f"{prefix}_pmg_occupancy": occupancy,
                f"{prefix}_pmg_candidates": lg,
                f"{prefix}_loss_scale": world_size * ((1 - target_g) * local_static_mass + target_g * lg / ng.clamp_min(1)),
            })
            if sharded_sources:
                self.sampling_metrics.update({
                    f"{prefix}_pmg_target_fraction": target_g,
                    f"{prefix}_loss_scale": world_size * (
                        (1 - target_g) * local_static_mass / static_rank_count.clamp_min(1)
                        + target_g * lg / ng.clamp_min(1)),
                })
            self._sampling_cache[cache_name] = (key, cdf, refs, importance)
        self._require_mixed_support(cdf[-1] > 0, f"no eligible mixed {kind} candidates on at least one rank")
        uniform = torch.rand(B, device=self.device, dtype=cdf.dtype, generator=generator)
        draws = torch.minimum(uniform * cdf[-1], torch.nextafter(cdf[-1], cdf.new_zeros(())))
        logical = torch.searchsorted(cdf, draws, right=True)
        return logical, importance[logical]

    def _mixed_clip_mask(self, out, motion):
        static = (out["ref_id"] >= 0) & ~out["generated_source"]
        positions = static.flatten().nonzero(as_tuple=False).flatten()
        valid = torch.ones_like(static)
        clips = torch.full_like(out["ref_id"], -1)
        if positions.numel():
            frames = out["f"].flatten()[positions]
            valid.flatten()[positions] = frames < self._motion_clip_end(motion, frames)
            clips.flatten()[positions] = motion.clip_id[frames]
        if static.shape[1] > 1:
            valid[:, 1:] &= ~(static[:, 1:] & static[:, :-1]) | (clips[:, 1:] == clips[:, :-1])
        return valid.float().cumprod(1)

    @torch.no_grad()
    def sample_windows(
        self,
        B: int,
        k: int,
        generator: torch.Generator | None = None,
        *,
        motion: object | None = None,
        bin_probabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample ``B`` contiguous ``k``-transition windows.

        A terminal transition is valid; transitions after it are masked. Ages are
        always drawn from the current logical range, so a window cannot splice an
        overwritten ring slot into its future. Perception fields have the same
        [B, k] leading dimensions as s/s1; the *1 fields are pre-reset targets.
        Optional bin probabilities weight eligible starts, with a per-candidate
        cap of eight times local uniform. This path also masks clip boundaries
        and returns [B] sampling_weight for the per-window loss before averaging.
        """
        if B <= 0 or k <= 0:
            raise ValueError(f"B and k must be positive, got {B}, {k}")
        if self.reference is not None:
            self._require_mixed_support(self.size >= k, "mixed replay window unavailable on at least one rank")
        if self.size < k:
            raise RuntimeError(
                f"replay has {self.size} transitions but a {k}-transition "
                f"window is required"
            )
        if self.reference is not None:
            logical, mixed_weight = self._mixed_logical(B, "windows", k, 0, motion, bin_probabilities, generator)
            t0, env = logical // self.N, logical % self.N
        elif bin_probabilities is not None:
            logical, importance = self._weighted_logical(B, "windows", k, motion, bin_probabilities, generator)
            t0, env = logical // self.N, logical % self.N
        elif self.recurrent:
            ages0 = torch.arange(self.size - k + 1, device=self.device)
            episode_steps = self.data["ep_step"][self._rows_for_ages(ages0)]
            eligible = (episode_steps <= ages0[:, None]) | (ages0[:, None] >= self.burn_in)
            logical = torch.multinomial(eligible.flatten().float(), B, replacement=True, generator=generator)
            t0, env = logical // self.N, logical % self.N
        else:
            t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device, generator=generator)
            env = torch.randint(0, self.N, (B,), device=self.device, generator=generator)
        ages = t0[:, None] + torch.arange(k, device=self.device)[None]
        rows = self._rows_for_ages(ages)
        envs = env[:, None].expand_as(rows)
        fields = ["s", "s1", "a", "a_prev", "xi", "f", "ep_step", "done"]
        if self.reference is not None:
            fields.extend(("ref_id", "ref_phase", "scene_id", "generated_source", "policy_obs"))
        if self.recovery_enabled:
            fields.append("recovery_role")
        if self.perception:
            fields.extend(self._perception_fields)
        if self.recurrent:
            fields.extend(("clean_depth", "clean_depth1"))
        out = self._gather(tuple(fields), rows, envs)
        if self.recurrent:
            out["burnin"] = self._burnin(t0, env, out["ep_step"][:, 0])

        # Transition j is alive iff none of transitions 0..j-1 ended an episode.
        if k == 1:
            mask = torch.ones((B, 1), dtype=torch.float32, device=self.device)
        else:
            prefix = torch.cumprod(1.0 - out["done"][:, :-1].float(), dim=1)
            mask = torch.cat(
                (torch.ones((B, 1), dtype=prefix.dtype, device=self.device), prefix),
                dim=1,
            )
        if self.reference is not None:
            out["sampling_weight"] = mixed_weight
            if bin_probabilities is not None:
                mask *= self._mixed_clip_mask(out, motion)
        elif bin_probabilities is not None:
            out["sampling_weight"] = importance[motion.frame_bin[out["f"][:, 0]]]
            clips = motion.clip_id[out["f"]]
            end = motion.clip_end[clips[:, :1]]
            in_clip = (clips == clips[:, :1]) & (out["f"] < end)
            mask *= in_clip.float().cumprod(dim=1)
        out["mask"] = mask
        return out

    def _motion_clip_end(
        self, motion: object, frames: torch.Tensor
    ) -> torch.Tensor:
        clip_id = getattr(motion, "clip_id", None)
        clip_end = getattr(motion, "clip_end", None)
        if not torch.is_tensor(clip_id) or not torch.is_tensor(clip_end):
            raise TypeError("motion must expose tensor clip_id and clip_end")
        if clip_id.device != self.device or clip_end.device != self.device:
            raise RuntimeError("motion.clip_id and motion.clip_end must be on the replay device")
        if clip_id.ndim != 1 or clip_end.ndim != 1:
            raise ValueError("motion.clip_id and motion.clip_end must be rank one")
        # Tensor indexing reports invalid simulator-provided indices without a
        # scalar read; range reductions here would synchronize the hot path.
        return clip_end[clip_id[frames]]

    @torch.no_grad()
    def sample_starts(
        self,
        B: int,
        hist: int,
        horizon: int,
        motion: object,
        generator: torch.Generator | None = None,
        *,
        bin_probabilities: torch.Tensor | None = None,
        decision_only: bool = False,
        decision_weight: torch.Tensor | float | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample policy seeds and padded observation histories.

        The newest history element is the seed state. Going backward is clamped by
        both the current episode length and available ring age; a short episode
        therefore repeats its first state rather than reading a previous episode
        or an overwritten slot. Legacy seeds leave at least ``horizon`` reference
        frames inside their clip. Weighted seeds only require ``f < clip_end``;
        the caller MUST mask imagination and critic targets at that clip end.
        Their [B] sampling_weight multiplies each actor return and critic MSE
        before averaging; do not also multiply by sampling_scale.
        decision_weight supplies an already aggregated per-rank decision weight;
        the caller then owns the global count collective and empty-rank routing.
        """
        if B <= 0 or hist <= 0 or horizon < 0:
            raise ValueError(
                f"B and hist must be positive and horizon non-negative, "
                f"got {B}, {hist}, {horizon}"
            )
        if self.reference is not None and not decision_only:
            self._require_mixed_support(self.size > 0, "mixed replay is empty on at least one rank")
        if self.size == 0:
            if decision_only and self.reference is not None:
                raise NoPMGDecisionSeeds("no generated decision seeds; replay is empty")
            raise RuntimeError(
                "replay is empty after cold start/resume; collect warmup transitions first"
            )

        if decision_only and self.reference is None:
            raise RuntimeError("decision_only requires attach_pmg")
        if self.reference is not None:
            logical, mixed_weight = self._mixed_logical(B, "starts", 1, horizon, motion, bin_probabilities, generator,
                                                        decision_only=decision_only, decision_weight=decision_weight)
        elif bin_probabilities is not None:
            logical, importance = self._weighted_logical(B, "starts", 1, motion, bin_probabilities, generator)
        else:
            ages = torch.arange(self.size, device=self.device)
            rows = self._rows_for_ages(ages)
            envs = torch.arange(self.N, device=self.device)[None, :].expand(
                self.size, self.N
            )
            candidate_rows = rows[:, None].expand(self.size, self.N)
            frames = self.data["f"][candidate_rows, envs]
            clip_end = self._motion_clip_end(motion, frames)
            eligible = frames + horizon <= clip_end
            if self.recurrent:
                episode_steps = self.data["ep_step"][candidate_rows, envs]
                required = max(self.burn_in, self.camera_capacity - 1)
                eligible &= (episode_steps <= ages[:, None]) | (ages[:, None] >= required)
            eligible = eligible.reshape(-1).to(torch.float32)

            # Reusing eligible candidates is GPU-side and avoids an unbounded Python
            # rejection loop or a data-dependent nonzero allocation.
            try:
                logical = torch.multinomial(
                    eligible, B, replacement=True, generator=generator
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"no replay start has {horizon} reference frames left before its clip end"
                ) from exc
        age = logical // self.N
        env = logical % self.N
        seed_rows = self._rows_for_ages(age)
        fields = ("s", "a_prev", "f", "xi", "ep_step")
        if self.reference is not None:
            fields += ("generated_source", "policy_obs")
        if self.recovery_enabled:
            fields += ("recovery_role",)
        seed = self._gather(fields + ("ref_id", "ref_phase", "scene_id"), seed_rows, env)
        if self.reference is not None:
            seed["sampling_weight"] = mixed_weight
            if decision_only:
                seed.update(self._gather(("decision_ref_id", "decision_ref_phase", "pmg_decision", "task_command"),
                                         seed_rows, env))
                seed["command"] = seed["task_command"]
        elif bin_probabilities is not None:
            seed["sampling_weight"] = importance[motion.frame_bin[seed["f"]]]
        if self.perception and (not self.recurrent or self.reference is not None):
            # The seed's own depth frame, so imagination starts from the latent
            # of the frame the policy actually saw at that step.
            seed.update(self._gather(("depth",), seed_rows, env))
        if self.recurrent:
            seed.update(self._gather(("clean_depth", "scan", "hit", "camera_lag"), seed_rows, env))
            seed["burnin"] = self._burnin(age, env, seed["ep_step"])
            camera_back = torch.arange(self.camera_capacity - 1, -1, -1, device=self.device)
            camera_back = torch.minimum(camera_back[None], torch.minimum(age, seed["ep_step"].long())[:, None])
            camera_rows = self._rows_for_ages(age[:, None] - camera_back)
            seed["camera_frames"] = self.data["camera_frame"][camera_rows, env[:, None]]
            if decision_only:
                capture_step = ((seed["ep_step"].long() - seed["camera_lag"].long()).clamp_min(0)
                                // self._pmg_capture_steps) * self._pmg_capture_steps
                capture_age = seed["ep_step"].long() - capture_step
                captured = self._rows_for_ages(age - capture_age)
                seed["capture_state"] = self.data["s"][captured, env]
                seed["capture_age"] = capture_age

        back = torch.arange(hist - 1, -1, -1, device=self.device, dtype=torch.long)
        ep_step = seed["ep_step"].long()
        back = torch.minimum(back[None, :], ep_step[:, None])
        back = torch.minimum(back, age[:, None])
        hist_rows = self._rows_for_ages(age[:, None] - back)
        hist_envs = env[:, None].expand_as(hist_rows)
        history = self._gather(fields, hist_rows, hist_envs)
        return seed, history
