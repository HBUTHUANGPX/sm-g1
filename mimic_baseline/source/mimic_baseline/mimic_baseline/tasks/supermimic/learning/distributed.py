"""Synchronous data-parallel primitives shared by the SuperMimic entry point.

This module deliberately contains no trainer state. It provides the small set
of collectives needed by the frozen-rollout/BPTT trainer: broadcasting the
shared learner, averaging complete gradients before clipping, pooling fresh
normalizer and sampler moments, and moving rank-local RNG state through rank 0.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterable, Sequence
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


def default_lr_scale(world_size: int) -> float:
    """Keep the base learning rate unchanged at every supported rank count."""
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return 1.0


def resolve_device(
    requested: str | torch.device, *, local_rank: int, world_size: int
) -> str:
    """Resolve one process to its CUDA device without sharing another rank's GPU."""
    if local_rank < 0:
        raise ValueError(f"local_rank must be non-negative, got {local_rank}")
    device = torch.device(requested)
    if world_size > 1 and device.type == "cuda":
        return f"cuda:{local_rank}"
    return str(device)


class DistributedContext:
    """Process-group wrapper for one synchronous worker.

    ``world_size == 1`` is a real no-op context: no process group is initialized
    and the existing single-card code path remains collective-free.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_rank: int,
        device: str | torch.device,
        backend: str | None = None,
        init_method: str | None = None,
        timeout_seconds: int = 1800,
        initialize: bool = True,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside [0, {world_size})")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.device = torch.device(
            resolve_device(
                device, local_rank=self.local_rank, world_size=self.world_size
            )
        )
        self.backend = backend
        self._owns_group = False
        self._init_method = init_method
        self._timeout_seconds = int(timeout_seconds)
        if initialize:
            self.initialize()

    def initialize(self) -> None:
        """Initialize the process group once, after any simulator launch setup."""
        if not self.enabled or self._owns_group:
            return
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        selected_backend = self.backend or (
            "nccl" if self.device.type == "cuda" else "gloo"
        )
        selected_init = self._init_method
        if selected_init is None and not (
            os.environ.get("RANK") is not None
            and os.environ.get("WORLD_SIZE") is not None
        ):
            raise RuntimeError(
                "multiple ranks require torchrun environment or init_method"
            )
        dist.init_process_group(
            backend=selected_backend,
            init_method=selected_init,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=self._timeout_seconds),
            **({"device_id": self.device} if self.device.type == "cuda" else {}),
        )
        self.backend = selected_backend
        self._owns_group = True

    @classmethod
    def from_environment(
        cls,
        *,
        device: str | torch.device,
        backend: str | None = None,
        timeout_seconds: int = 1800,
        initialize: bool = True,
    ) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            backend=backend,
            timeout_seconds=timeout_seconds,
            initialize=initialize,
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self._owns_group and dist.is_initialized():
            dist.destroy_process_group()
        self._owns_group = False

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def broadcast_module_state(self, modules: Iterable[nn.Module]) -> None:
        """Broadcast parameters and buffers from the single shared learner."""
        if not self.enabled:
            return
        with torch.no_grad():
            for module in modules:
                for tensor in tuple(module.parameters()) + tuple(module.buffers()):
                    dist.broadcast(tensor.data, src=0)

    def average_gradients(self, modules: Iterable[nn.Module]) -> None:
        """Sum rank-local gradients, divide once, and copy the shared average back."""
        if not self.enabled:
            return
        gradients = [
            parameter.grad
            for module in modules
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return
        devices = {gradient.device for gradient in gradients}
        dtypes = {gradient.dtype for gradient in gradients}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("flat gradient averaging requires one device and dtype")
        flat = torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)
        offset = 0
        with torch.no_grad():
            for gradient in gradients:
                count = gradient.numel()
                gradient.copy_(flat[offset : offset + count].view_as(gradient))
                offset += count

    def update_running_norms(
        self,
        normalizers: Sequence[Any],
        samples: Sequence[torch.Tensor],
    ) -> None:
        """Update running statistics from globally pooled fresh sample moments.

        All local sufficient statistics are packed into one collective. Running
        standard deviations are never averaged; they are recomputed from pooled
        counts, means, and second moments before touching the running buffers.
        """
        if not normalizers:
            return
        if not self.enabled:
            for normalizer, sample in zip(normalizers, samples, strict=True):
                normalizer.update(sample)
            return

        sums: list[torch.Tensor] = []
        square_sums: list[torch.Tensor] = []
        counts: list[torch.Tensor] = []
        for normalizer, sample in zip(normalizers, samples, strict=True):
            value = (
                sample.detach()
                .reshape(-1, normalizer.dim)
                .to(
                    device=normalizer.mean.device,
                    dtype=normalizer.mean.dtype,
                )
            )
            sums.append(value.sum(0))
            square_sums.append(value.square().sum(0))
            counts.append(
                torch.tensor(
                    [value.shape[0]],
                    device=normalizer.mean.device,
                    dtype=normalizer.mean.dtype,
                )
            )
        packed = torch.cat((*sums, *square_sums, *counts))
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        offset = 0
        moments: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        sum_offset = 0
        square_offset = sum(value.numel() for value in sums)
        count_offset = square_offset + sum(value.numel() for value in square_sums)
        for normalizer in normalizers:
            dim = normalizer.dim
            count = packed[count_offset + offset]
            denominator = count.clamp_min(1.0)
            mean = packed[sum_offset : sum_offset + dim] / denominator
            square_mean = packed[square_offset : square_offset + dim] / denominator
            variance = (square_mean - mean.square()).clamp_min(0.0)
            moments.append((mean, variance, count))
            sum_offset += dim
            square_offset += dim
            offset += 1
        for normalizer, (mean, variance, count) in zip(
            normalizers, moments, strict=True
        ):
            normalizer.update_moments(mean, variance, count)

    def all_gather_objects(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def sum_and_count_tensors(
        self,
        values: Sequence[torch.Tensor],
        counts: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Sum metric numerators and denominators with one device collective."""
        if len(values) != len(counts):
            raise ValueError("metric values and counts must have equal length")
        if not values:
            return [], []
        packed = torch.cat(
            tuple(
                value.detach()
                .reshape(1)
                .to(
                    device=self.device,
                    dtype=torch.float32,
                )
                for value in (*values, *counts)
            )
        )
        if not self.enabled:
            split = len(values)
            return (
                list(packed[:split]),
                list(packed[split:]),
            )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        split = len(values)
        return (
            list(packed[:split]),
            list(packed[split:]),
        )

    def rank_rng_state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "torch": torch.get_rng_state().cpu(),
            "python": random.getstate(),
        }
        if self.device.type == "cuda":
            payload["cuda"] = torch.cuda.get_rng_state(self.device).cpu()
        return payload

    def gather_rank_rng_states(self) -> list[dict[str, Any]]:
        return self.all_gather_objects(self.rank_rng_state())

    def restore_checkpoint_rng(self, checkpoint: dict[str, Any]) -> None:
        """Restore this rank's sampler RNG, including one-checkpoint expansion."""
        states = checkpoint.get("rng_by_rank")
        legacy = checkpoint.get("rng")
        if states:
            exact = self.rank < len(states)
            source = states[self.rank] if exact else states[0]
        elif legacy:
            exact = self.rank == 0
            source = legacy
        else:
            return

        cpu_state = source.get("torch")
        if cpu_state is None:
            return
        if exact:
            torch.set_rng_state(cpu_state.cpu())
            if source.get("python") is not None:
                random.setstate(source["python"])
            if self.device.type == "cuda" and source.get("cuda") is not None:
                _restore_cuda_rng(source["cuda"], self.device)
            return

        seed = _derived_seed(cpu_state, self.rank)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        torch.set_rng_state(generator.get_state())
        random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)


def _derived_seed(state: torch.Tensor, rank: int) -> int:
    digest = hashlib.sha256()
    digest.update(state.detach().cpu().numpy().tobytes())
    digest.update(f":ddp-rank:{rank}".encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def _restore_cuda_rng(state: Any, device: torch.device) -> None:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if isinstance(state, (list, tuple)):
        if index >= len(state):
            raise ValueError(
                f"checkpoint has {len(state)} CUDA RNG states, cannot restore cuda:{index}"
            )
        state = state[index]
    torch.cuda.set_rng_state(state.cpu(), device=device)


__all__ = [
    "DistributedContext",
    "default_lr_scale",
    "resolve_device",
]
