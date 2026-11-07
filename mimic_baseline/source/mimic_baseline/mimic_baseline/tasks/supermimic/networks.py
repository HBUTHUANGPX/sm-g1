"""World model, motion policy, and critic from SuperMimic Appendix C.

The model and optimizer code in this package is derived in part from the
MIT-licensed implementation at commit 27126bddd74381d164118ef83ee78edb58484a6b.
The authoritative behavior is the local SuperMimic RX-P2 contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .mdp_legacy import ACT_DIM, INV_DIM, NUM_J, XI_DIM, RELATIVE_ORIENTATION_DIM
from .motion import CONTROL_DT
from .perception import DEPTH_CLIP_M, HEIGHT_SCAN_DIM, LATENT_DIM, DepthEncoder, HeightEncoder
from .recovery import recovery_yaw_mat
from .rotations import mat_to_6d, rot6d_to_mat, yaw_mat


class RunningNorm(nn.Module):
    """Running per-feature normalizer with source-compatible statistics.

    ``count`` is a float buffer so it is serialized with the model. Statistics
    stop changing at ``max_count``; the freeze decision is tensor-side, avoiding
    an implicit device synchronization in the update path.
    """

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"RunningNorm dim must be positive, got {dim}")
        self.dim = int(dim)
        self.max_count = float(max_count)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.dim))
        self.register_buffer("var", torch.ones(self.dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, self.dim).to(dtype=self.mean.dtype)
        n = x.shape[0]

        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        self.update_moments(batch_mean, batch_var, n)

    @torch.no_grad()
    def update_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int | torch.Tensor,
    ) -> None:
        """Merge already-pooled fresh moments into the running statistics."""
        batch_mean = batch_mean.detach().to(
            device=self.mean.device, dtype=self.mean.dtype
        )
        batch_var = batch_var.detach().to(
            device=self.mean.device, dtype=self.mean.dtype
        )
        n = torch.as_tensor(
            batch_count, device=self.mean.device, dtype=self.mean.dtype
        )
        total = self.count + n
        delta = batch_mean - self.mean
        candidate_mean = self.mean + delta * (n / total)
        candidate_var = (
            self.var * self.count
            + batch_var * n
            + delta.square() * (self.count * n / total)
        ) / total
        # Match the source's "stop after a batch that crosses max_count" rule.
        active = self.count < self.max_count
        self.mean.copy_(torch.where(active, candidate_mean, self.mean))
        self.var.copy_(torch.where(active, candidate_var, self.var))
        self.count.copy_(torch.where(active, total, self.count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


def _mlp(
    input_dim: int,
    hidden: Sequence[int],
    output_dim: int,
    *,
    activation: type[nn.Module],
    layer_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for next_width in hidden:
        layers.append(nn.Linear(width, next_width))
        if layer_norm:
            layers.append(nn.LayerNorm(next_width))
        layers.append(activation())
        width = next_width
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


# p(3) + rot6d increment(6) + v(3) + w(3) + q(29) + qd(29)
WM_OUT = 3 + 6 + 3 + 3 + NUM_J + NUM_J
# 6-D encoding of the identity rotation in ``mat_to_6d`` order, which flattens
# the first two matrix columns row-major: [R00, R01, R10, R11, R20, R21].
# The network regresses the residual around it so a zero output means "no change".
IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


class WorldModel(nn.Module):
    """Predict frame-local state increments from state, action pair, and xi.

    Position and linear/angular velocity are expressed in heading/body frames so
    the network never learns absolute world translation. Rotation is predicted as
    a 6-D increment around the identity and projected back onto SO(3), which is
    continuous everywhere (unlike a rotation vector at pi).

    ``a_{t-1}`` is part of the input because the actuator applies each command
    after 0-4 physics substeps (0-20 ms), i.e. at most one control step of delay.
    The transition therefore depends on the two most recent actions.
    """

    def __init__(
        self,
        hidden: Sequence[int] = (2048, 2048, 2048),
        target_mode: str = "residual",
        latent_dim: int = 0,
        *,
        recovery_enabled: bool = False,
        action_dim: int = ACT_DIM,
    ):
        super().__init__()
        if target_mode not in ("residual", "increment"):
            raise ValueError(
                f"target_mode must be 'residual' or 'increment', got {target_mode!r}"
            )
        if latent_dim < 0:
            raise ValueError(f"latent_dim must be non-negative, got {latent_dim}")
        self.target_mode = target_mode
        self._heading = recovery_yaw_mat if recovery_enabled else yaw_mat
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        if self.action_dim not in (27, 29):
            raise ValueError("Supported action dimensions are RX-P2 27 and G1 29")
        self.in_dim = INV_DIM + 2 * self.action_dim + XI_DIM + self.latent_dim
        self.out_dim = WM_OUT + self.latent_dim
        self.net = _mlp(
            self.in_dim,
            hidden,
            self.out_dim,
            activation=nn.SiLU,
            layer_norm=True,
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(
        self,
        S: tuple[torch.Tensor, ...],
        a: torch.Tensor,
        a_prev: torch.Tensor,
        xi_n: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from .mdp_legacy import invariant_features

        features = torch.cat((invariant_features(S, heading=self._heading(S[1])), a, a_prev, xi_n), dim=-1)
        if self.latent_dim:
            if latent is None or latent.shape[-1] != self.latent_dim:
                got = None if latent is None else tuple(latent.shape)
                raise ValueError(
                    f"latent must end in {self.latent_dim} for perception "
                    f"conditioning, got {got}"
                )
            features = torch.cat((features, latent), dim=-1)
        elif latent is not None:
            raise ValueError("this world model was built without a perception latent")
        return features

    def targets(
        self,
        S: tuple[torch.Tensor, ...],
        S1: tuple[torch.Tensor, ...],
        latent: torch.Tensor | None = None,
        latent1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the frame-local regression target from S to S1.

        ``target_mode="residual"`` (the default) puts position and joints on top
        of explicit integration, ``p' = p + dt*v' + R_h*dp`` and
        ``q' = q + dt*qd' + dq``, so a zero network output reproduces
        constant-velocity extrapolation instead of a zero increment. That is the
        Reproduced parameterization and gives the fit a physical prior.

        ``target_mode="increment"`` is the plain-delta control arm: it predicts
        the whole per-step displacement and has no such prior. It exists only so
        the two parameterizations can be compared under identical data.

        Rotation keeps the 6-D residual around identity in both modes, projected
        back onto SO(3) in :meth:`step`.

        With a perception latent the target grows by the latent's own residual
        ``z1 - z``. An identity prior is right here for the same reason it is
        right for the physical state: two consecutive camera frames at 50 Hz are
        far more alike than a fresh frame is to zero.
        """
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        target = self.physical_targets(S, S1)
        if not self.latent_dim:
            if latent is not None or latent1 is not None:
                raise ValueError("this world model was built without a perception latent")
            return target
        if latent is None or latent1 is None:
            raise ValueError(
                "a perception world model requires both latent and latent1 targets"
            )
        return torch.cat((target, latent1 - latent), dim=-1)

    def physical_targets(
        self,
        S: tuple[torch.Tensor, ...],
        S1: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return only the 73-D physical part, without the latent residual.

        Diagnostics still need a physical-only error when the perception branch
        is active; routing that through :meth:`targets` would force a sentinel
        latent and mix a meaningless latent term into the report.
        """
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        heading = self._heading(R)
        body_from_world = R.transpose(-1, -2)
        dt = CONTROL_DT if self.target_mode == "residual" else 0.0
        return torch.cat(
            (
                (
                    heading.transpose(-1, -2)
                    @ ((p1 - p) - dt * v1).unsqueeze(-1)
                ).squeeze(-1),
                mat_to_6d(body_from_world @ R1) - IDENTITY_6D.to(R1.device),
                (body_from_world @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (body_from_world @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                (q1 - q) - dt * qd1,
                qd1 - qd,
            ),
            dim=-1,
        )

    def step(
        self,
        S: tuple[torch.Tensor, ...],
        a: torch.Tensor,
        a_prev: torch.Tensor,
        xi_n: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Advance one control step.

        Returns the next physical state. With a perception latent the result is
        ``(S1, z1)``, because ``z`` is a rollout state alongside the 73-D state
        and must be threaded through imagination exactly like it.
        """
        x = self.in_norm(self.inputs(S, a, a_prev, xi_n, latent))
        normalized_delta = self.net(x)
        delta = normalized_delta * self.out_norm.std() + self.out_norm.mean
        next_state = self._physical_step(S, delta[..., :WM_OUT])
        if not self.latent_dim:
            return next_state
        if latent is None:
            raise ValueError("a perception world model requires the current latent")
        return next_state, latent + delta[..., WM_OUT:]

    def _physical_step(
        self,
        S: tuple[torch.Tensor, ...],
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Apply the shared 73-D physical transition for all WM variants."""
        p, R, v, w, q, qd = S
        dp, dr6, dv, dw, dq, dqd = delta.split(
            (3, 6, 3, 3, NUM_J, NUM_J), dim=-1
        )
        heading = self._heading(R)
        dt = CONTROL_DT if self.target_mode == "residual" else 0.0
        # Velocity first, then integrate position with the *updated* velocity:
        # this mirrors :meth:`targets`, so a zero residual reproduces
        # constant-velocity extrapolation.
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        qd1 = qd + dqd
        return (
            p + dt * v1 + (heading @ dp.unsqueeze(-1)).squeeze(-1),
            R @ rot6d_to_mat(IDENTITY_6D.to(R.device) + dr6),
            v1,
            w1,
            q + dt * qd1 + dq,
            qd1,
        )

    def state_error(
        self,
        S_hat: tuple[torch.Tensor, ...],
        S: tuple[torch.Tensor, ...],
        latent_hat: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return same-time prediction errors in the reference state's frames.

        The layout matches the targets, but neither target mode integrates here:
        subtracting dt*v_hat or dt*qd_hat would penalize an exact moving-state
        prediction. Those priors belong only to physical_targets and step.

        Without a latent pair only the physical part is returned, even on a
        perception model. Diagnostics report physical error separately, and a
        sentinel latent there would inject a meaningless term into the metric.
        """
        p, R, v, w, q, qd = S
        p_hat, R_hat, v_hat, w_hat, q_hat, qd_hat = S_hat
        heading = self._heading(R)
        body_from_world = R.transpose(-1, -2)
        error = torch.cat(
            (
                (heading.transpose(-1, -2) @ (p_hat - p).unsqueeze(-1)).squeeze(-1),
                mat_to_6d(body_from_world @ (R_hat - R)),
                (body_from_world @ (v_hat - v).unsqueeze(-1)).squeeze(-1),
                (body_from_world @ (w_hat - w).unsqueeze(-1)).squeeze(-1),
                q_hat - q,
                qd_hat - qd,
            ),
            dim=-1,
        )
        if self.latent_dim and latent_hat is not None and latent is not None:
            return torch.cat((error, latent_hat - latent), dim=-1)
        return error


class RecurrentWorldModel(WorldModel):
    """Predict 73 physical increments and a raw 64-D recurrent driver.

    Reuses the physical-only input, target, error and integration contracts of
    ``WorldModel``. Current height/depth features condition the transition; the
    driver is learned through recurrent reconstruction, never normalized by
    physical target statistics or accumulated as a latent residual.
    """

    def __init__(
        self,
        hidden: Sequence[int] = (2048, 2048, 2048),
        target_mode: str = "residual",
        *,
        recovery_enabled: bool = False,
    ):
        nn.Module.__init__(self)
        if target_mode not in ("residual", "increment"):
            raise ValueError(
                f"target_mode must be 'residual' or 'increment', got {target_mode!r}"
            )
        self.target_mode = target_mode
        self._heading = recovery_yaw_mat if recovery_enabled else yaw_mat
        # Keep inherited targets/state_error strictly physical-only.
        self.latent_dim = 0
        self.feature_dim = 2 * LATENT_DIM
        self.driver_dim = LATENT_DIM
        self.in_dim = INV_DIM + ACT_DIM + ACT_DIM + XI_DIM + self.feature_dim
        self.out_dim = WM_OUT + self.driver_dim
        self.net = _mlp(
            self.in_dim,
            hidden,
            self.out_dim,
            activation=nn.SiLU,
            layer_norm=True,
        )
        # A nonzero driver head keeps input/trunk gradients alive on step one.
        nn.init.zeros_(self.net[-1].weight[:WM_OUT])
        nn.init.zeros_(self.net[-1].bias[:WM_OUT])
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    def inputs(
        self,
        S: tuple[torch.Tensor, ...],
        a: torch.Tensor,
        a_prev: torch.Tensor,
        xi_n: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must end in {self.feature_dim}, got {tuple(features.shape)}"
            )
        return torch.cat((super().inputs(S, a, a_prev, xi_n), features), dim=-1)

    def step(
        self,
        S: tuple[torch.Tensor, ...],
        a: torch.Tensor,
        a_prev: torch.Tensor,
        xi_n: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        output = self.net(self.in_norm(self.inputs(S, a, a_prev, xi_n, features)))
        physical = output[..., :WM_OUT] * self.out_norm.std() + self.out_norm.mean
        return self._physical_step(S, physical), output[..., WM_OUT:]


class Policy(nn.Module):
    """Motion encoder and shared deterministic 27-action decoder."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        hidden: Sequence[int] = (2048, 1024, 512, 256),
        perception_dim: int = 0,
        *,
        actor_relative_orientation: bool = False,
        action_dim: int = ACT_DIM,
    ):
        super().__init__()
        if obs_dim <= 0 or goal_dim <= 0 or hist <= 0 or z_dim <= 0:
            raise ValueError("Policy dimensions and history length must be positive")
        if perception_dim < 0:
            raise ValueError(
                f"perception_dim must be non-negative, got {perception_dim}"
            )
        self.action_dim = int(action_dim)
        if self.action_dim not in (27, 29):
            raise ValueError("Supported action dimensions are RX-P2 27 and G1 29")
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.hist = int(hist)
        self.z_dim = int(z_dim)
        self.perception_dim = int(perception_dim)
        self.relative_orientation_dim = RELATIVE_ORIENTATION_DIM if actor_relative_orientation else 0
        self.encoder = _mlp(
            self.goal_dim,
            hidden,
            self.z_dim,
            activation=nn.ELU,
            layer_norm=False,
        )
        self.decoder = _mlp(
            self.z_dim + self.obs_dim * self.hist + self.perception_dim + self.relative_orientation_dim,
            hidden,
            self.action_dim,
            activation=nn.ELU,
            layer_norm=False,
        )
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.01)
            self.decoder[-1].bias.zero_()
        self.obs_norm = RunningNorm(self.obs_dim)
        self.goal_norm = RunningNorm(self.goal_dim)
        self.perception_norm = (
            RunningNorm(self.perception_dim) if self.perception_dim else None
        )

    def forward(
        self,
        obs_hist: torch.Tensor,
        goal: torch.Tensor,
        perception: torch.Tensor | None = None,
        relative_orientation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode an action from proprioceptive history, goal, and perception.

        ``perception`` is the current depth latent. It enters the decoder
        directly rather than being averaged into the history: the latents of
        consecutive frames are nearly identical, so the newest one carries the
        current terrain geometry and the older ones add little.

        In imagination the caller passes the *predicted* latent. Feeding a
        recorded one would leak the true future terrain into the rollout and
        make the actor's imagined return optimistic in exactly the part of the
        state the world model predicts worst.
        """
        latent = torch.tanh(self.encoder(self.goal_norm(goal)))
        observations = self.obs_norm(obs_hist).flatten(-2)
        parts = [latent, observations]
        if self.perception_dim:
            if perception is None:
                raise ValueError(
                    f"this policy was built with perception_dim={self.perception_dim} "
                    "and requires a perception latent"
                )
            if perception.shape[-1] != self.perception_dim:
                raise ValueError(
                    f"perception must end in {self.perception_dim}, "
                    f"got {tuple(perception.shape)}"
                )
            parts.append(self.perception_norm(perception))
        elif perception is not None:
            raise ValueError("this policy was built without perception")
        if self.relative_orientation_dim:
            if relative_orientation is None or relative_orientation.shape[-1] != self.relative_orientation_dim:
                raise ValueError("this policy requires relative_orientation[...,6]")
            # Current feedback stays outside history and the reference encoder.
            parts.append(relative_orientation)
        elif relative_orientation is not None:
            raise ValueError("this policy was built without relative_orientation")
        return self.decoder(torch.cat(parts, dim=-1))


class DepthPolicy(Policy):
    """Policy with an independently trained 2 m encoder of raw ERP frames."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        hidden: Sequence[int] = (2048, 1024, 512, 256),
        perception_dim: int = LATENT_DIM,
        *,
        actor_relative_orientation: bool = False,
    ):
        super().__init__(obs_dim, goal_dim, hist, z_dim, hidden, perception_dim,
                         actor_relative_orientation=actor_relative_orientation)
        self.depth_encoder = DepthEncoder(self.perception_dim, clip_m=DEPTH_CLIP_M)

    def encode_perception(
        self, depth: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.depth_encoder(depth, valid)

    def forward(
        self,
        obs_hist: torch.Tensor,
        goal: torch.Tensor,
        depth: torch.Tensor,
        relative_orientation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(obs_hist, goal, self.encode_perception(depth), relative_orientation)


class Critic(nn.Module):
    """One value network per interface: paper Table A.4 (256, 128)."""

    def __init__(self, in_dim: int):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"Critic in_dim must be positive, got {in_dim}")
        self.in_dim = int(in_dim)
        self.net = _mlp(
            self.in_dim,
            (256, 128),
            1,
            activation=nn.ELU,
            layer_norm=False,
        )
        self.norm = RunningNorm(self.in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)


class HeightCritic(Critic):
    """Privileged critic accepting raw base, scan and hit inputs on each call."""

    def __init__(
        self, base_in_dim: int, *, world_frame: bool = False,
        scan_dim: int = HEIGHT_SCAN_DIM,
    ):
        if base_in_dim <= 0:
            raise ValueError(f"base_in_dim must be positive, got {base_in_dim}")
        super().__init__(base_in_dim + LATENT_DIM)
        self.base_in_dim = int(base_in_dim)
        self.scan_dim = scan_dim
        self.world_frame = bool(world_frame)
        self.in_dim = self.base_in_dim + 2 * self.scan_dim + (6 if self.world_frame else 0)
        self.height_encoder = HeightEncoder(world_frame=self.world_frame, scan_dim=scan_dim)

    def normalization_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"critic input must end in {self.in_dim}, got {tuple(x.shape)}")
        widths = (self.base_in_dim, self.scan_dim, self.scan_dim) + ((6,) if self.world_frame else ())
        parts = x.split(widths, dim=-1)
        base, scan, hit = parts[:3]
        orientation = parts[3] if self.world_frame else None
        return torch.cat((base, self.height_encoder(scan, hit, orientation)), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.normalization_features(x))
