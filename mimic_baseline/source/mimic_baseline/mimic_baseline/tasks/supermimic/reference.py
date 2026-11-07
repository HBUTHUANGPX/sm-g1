"""Structured, stateless reference sampling from a static motion catalog."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from .rotations import mat_to_6d, yaw_mat


@dataclass
class ReferenceState:
    p: torch.Tensor
    R: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    q: torch.Tensor
    qd: torch.Tensor
    v_local: torch.Tensor
    w_local: torch.Tensor
    body_p: torch.Tensor | None = None
    body_R: torch.Tensor | None = None
    body_v: torch.Tensor | None = None
    body_w: torch.Tensor | None = None

    @property
    def state(self) -> tuple[torch.Tensor, ...]:
        return self.p, self.R, self.v, self.w, self.q, self.qd

    @property
    def bodies(self) -> tuple[torch.Tensor | None, ...]:
        return self.body_p, self.body_R, self.body_v, self.body_w

    def detach(self) -> ReferenceState:
        return ReferenceState(
            *(
                value.detach() if value is not None else None
                for value in (*self.state, self.v_local, self.w_local, *self.bodies)
            )
        )


@dataclass
class ReferenceExpiry:
    episode_end: torch.Tensor
    replan_due: torch.Tensor
    remaining: torch.Tensor


@dataclass
class ReferenceBatch:
    now: ReferenceState
    future: ReferenceState | None
    valid: torch.Tensor
    expiry: ReferenceExpiry
    source_frame: torch.Tensor | None = None
    scene_id: torch.Tensor | None = None
    action_dim: int = 27

    def goal(
        self, mode: str = "reference", rotation: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.future is None:
            raise ValueError("Goal features require sampling with goal_frames > 0.")
        future = self.future
        if mode == "reference":
            heading_transpose = yaw_mat(self.now.R).transpose(-1, -2).unsqueeze(-3)
            features = torch.cat(
                (
                    (
                        heading_transpose
                        @ (future.p - self.now.p.unsqueeze(-2)).unsqueeze(-1)
                    ).squeeze(-1),
                    mat_to_6d(heading_transpose @ future.R),
                    future.v_local,
                    future.w_local,
                    future.q[..., :self.action_dim],
                    future.qd[..., :self.action_dim] * 0.1,
                ),
                dim=-1,
            )
        elif mode == "state_relative":
            if rotation is None:
                raise ValueError("state_relative goals require rotation.")
            relative_rotation = rotation.transpose(-1, -2).unsqueeze(-3) @ future.R
            features = torch.cat(
                (
                    future.q[..., :self.action_dim],
                    future.qd[..., :self.action_dim] * 0.1,
                    mat_to_6d(relative_rotation),
                ),
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown reference goal mode: {mode!r}.")
        return features.reshape(
            self.now.p.shape[:-1] + (features.shape[-2] * features.shape[-1],)
        )


def _sample_indices(catalog: Any, plan_ids, offsets) -> tuple[torch.Tensor, torch.Tensor]:
    device = catalog.clip_start.device
    return torch.broadcast_tensors(
        torch.as_tensor(plan_ids, dtype=torch.long, device=device),
        torch.as_tensor(offsets, dtype=torch.long, device=device),
    )


class StaticReference:
    """Sample a catalog without copying or modifying its resident tensor tables."""

    __slots__ = ("catalog",)

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog

    def locate(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames = torch.as_tensor(frames, dtype=torch.long, device=self.catalog.clip_start.device)
        clip_ids = self.catalog.clip_id[frames]
        return clip_ids, frames - self.catalog.clip_start[clip_ids]

    def _state(self, frames: torch.Tensor, *, bodies: bool = False) -> ReferenceState:
        catalog = self.catalog
        return ReferenceState(
            p=catalog.root_p[frames],
            R=catalog.root_R[frames],
            v=catalog.root_v[frames],
            w=catalog.root_w[frames],
            q=catalog.q[frames],
            qd=catalog.qd[frames],
            v_local=catalog.root_v_local[frames],
            w_local=catalog.root_w_local[frames],
            body_p=catalog.body_p[frames] if bodies else None,
            body_R=catalog.body_R[frames] if bodies else None,
            body_v=catalog.body_v[frames] if bodies else None,
            body_w=catalog.body_w[frames] if bodies else None,
        )

    def sample(
        self, plan_ids, offsets, goal_frames: int = 0, *, bodies: bool = False
    ) -> ReferenceBatch:
        if goal_frames < 0:
            raise ValueError(f"goal_frames must be nonnegative, got {goal_frames}.")
        plan_ids, offsets = _sample_indices(self.catalog, plan_ids, offsets)
        start = self.catalog.clip_start[plan_ids]
        last = self.catalog.clip_end[plan_ids] - start
        frames = start + torch.minimum(offsets.clamp_min(0), last)
        now = self._state(frames, bodies=bodies)
        future = None
        if goal_frames:
            steps = torch.arange(1, goal_frames + 1, device=offsets.device)
            future_frames = start[..., None] + torch.minimum(
                (offsets[..., None] + steps).clamp_min(0), last[..., None]
            )
            future = self._state(future_frames)
        return ReferenceBatch(
            action_dim=getattr(self.catalog, "action_dim", 27),
            now=now,
            future=future,
            valid=(offsets >= 0) & (offsets < last),
            expiry=ReferenceExpiry(
                episode_end=offsets >= last,
                replan_due=torch.zeros_like(offsets, dtype=torch.bool),
                remaining=(last - offsets).clamp_min(0),
            ),
            source_frame=frames,
            scene_id=plan_ids,
        )


class FixedReference:
    """Hold one source pose indefinitely, with zero reference velocities."""

    __slots__ = ("static_reference", "catalog", "frame")

    def __init__(self, static_reference: Any, frame: int) -> None:
        self.static_reference = (
            static_reference
            if isinstance(static_reference, StaticReference)
            else StaticReference(static_reference)
        )
        self.catalog = self.static_reference.catalog
        self.frame = int(frame)

    def locate(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.static_reference.locate(frames)

    def sample(
        self, plan_ids, offsets, goal_frames: int = 0, *, bodies: bool = False
    ) -> ReferenceBatch:
        if goal_frames < 0:
            raise ValueError(f"goal_frames must be nonnegative, got {goal_frames}.")
        _, offsets = _sample_indices(self.catalog, plan_ids, offsets)
        frames = torch.full_like(offsets, self.frame)
        sampled = self.static_reference._state(frames, bodies=bodies)
        now = replace(
            sampled,
            v=torch.zeros_like(sampled.v),
            w=torch.zeros_like(sampled.w),
            qd=torch.zeros_like(sampled.qd),
            v_local=torch.zeros_like(sampled.v_local),
            w_local=torch.zeros_like(sampled.w_local),
            body_v=torch.zeros_like(sampled.body_v) if sampled.body_v is not None else None,
            body_w=torch.zeros_like(sampled.body_w) if sampled.body_w is not None else None,
        )
        future = None
        if goal_frames:
            # Expand only root/joint values; future bodies are never materialized.
            future = ReferenceState(
                *(
                    value.unsqueeze(offsets.ndim).expand(
                        offsets.shape + (goal_frames,) + value.shape[offsets.ndim:]
                    )
                    for value in (*now.state, now.v_local, now.w_local)
                )
            )
        return ReferenceBatch(
            action_dim=getattr(self.catalog, "action_dim", 27),
            now=now,
            future=future,
            valid=torch.ones_like(offsets, dtype=torch.bool),
            expiry=ReferenceExpiry(
                episode_end=torch.zeros_like(offsets, dtype=torch.bool),
                replan_due=torch.zeros_like(offsets, dtype=torch.bool),
                remaining=torch.full_like(offsets, torch.iinfo(torch.long).max),
            ),
            source_frame=frames,
            scene_id=self.catalog.clip_id[frames],
        )


def ensure_static_reference(motion: Any) -> Any:
    if callable(getattr(motion, "sample", None)) and callable(getattr(motion, "locate", None)):
        return motion
    return StaticReference(motion)


__all__ = [
    "ReferenceState",
    "ReferenceExpiry",
    "ReferenceBatch",
    "StaticReference",
    "FixedReference",
    "ensure_static_reference",
]
