"""Analytic SuperMimic motion-tracking MDP for RX-P2."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import RXP2Kinematics
from .motion import CONTROL_DT, MotionLib
from .reference import ReferenceBatch, ensure_static_reference
from .rotations import mat_to_6d, rot6d_to_mat, rot_angle_sq, yaw_mat


ACT_DIM = 27
NUM_J = 29
# p(3) + rot6d(6) + v(3) + w(3) + q(29) + qd(29)
STATE_DIM = 73
OBS_DIM = 91
XI_DIM = 33
INV_DIM = 71
RELATIVE_ORIENTATION_DIM = 6
RELATIVE_ORIENTATION_NOISE = 0.05
CRITIC_EXTRA = 29 + 14 * 3 + 3 + 6 + XI_DIM
MAX_STEP_REWARD = 5.0 * CONTROL_DT


@dataclass
class RewardCfg:
    """Table A.2 widths, weights, limits, and termination thresholds."""

    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    # Paper Appendix A: root height error 0.25 m, gravity-z difference 0.8.
    term_height: float = 0.20
    term_tilt: float = 0.8


OBS_NOISE = torch.tensor(
    [0.01] * NUM_J
    + [0.5] * NUM_J
    + [0.2] * 3
    + [0.05] * 3
    + [0.0] * ACT_DIM
)


def flatten_state(
    state: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Pack ``(p,R,v,w,q,qd)`` into the 73-D state with a 6-D rotation."""
    p, rotation, velocity, angular_velocity, joint_pos, joint_vel = state
    return torch.cat(
        (
            p,
            mat_to_6d(rotation),
            velocity,
            angular_velocity,
            joint_pos,
            joint_vel,
        ),
        dim=-1,
    )


def unflatten_state(
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a 73-D state into pose, world velocity, and 29 joint tensors."""
    if packed.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected packed state width {STATE_DIM}, got {packed.shape[-1]}.")
    return (
        packed[..., 0:3],
        rot6d_to_mat(packed[..., 3:9]),
        packed[..., 9:12],
        packed[..., 12:15],
        packed[..., 15:44],
        packed[..., 44:73],
    )


def invariant_features(
    state: tuple[torch.Tensor, ...],
    *,
    heading: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the 33 heading- and position-invariant state features."""
    p, rotation, velocity, angular_velocity, joint_pos, joint_vel = state
    heading = yaw_mat(rotation) if heading is None else heading
    rotation_transpose = rotation.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(heading.transpose(-1, -2) @ rotation),
            (rotation_transpose @ velocity.unsqueeze(-1)).squeeze(-1),
            (rotation_transpose @ angular_velocity.unsqueeze(-1)).squeeze(-1),
            joint_pos,
            joint_vel,
        ),
        dim=-1,
    )


def relative_root_orientation(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Tracking-style reference orientation in the actual base_link frame."""
    return mat_to_6d(actual.transpose(-1, -2) @ reference)


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """Normalize Table A.3 domain-randomization parameters."""
    if xi.shape[-1] != XI_DIM:
        raise ValueError(f"Expected xi width {XI_DIM}, got {xi.shape[-1]}.")
    return torch.cat(
        (
            (xi[..., :1] - 0.75) / 0.45,
            xi[..., 1:30] / 0.01,
            xi[..., 30:33] / 0.05,
        ),
        dim=-1,
    )


class MDP:
    """Paper Appendix A observation, reward, termination, and critic features."""

    def __init__(
        self,
        kin: RXP2Kinematics,
        motion: MotionLib,
        default_q: torch.Tensor,
        cfg: RewardCfg = RewardCfg(),
        *,
        recovery_enabled: bool = False,
        reference=None,
    ) -> None:
        if default_q.shape[-1] != NUM_J:
            raise ValueError(f"default_q must end in {NUM_J}, got {tuple(default_q.shape)}.")
        self.kin = kin
        self.action_dim = getattr(kin, "action_dim", ACT_DIM)
        self.obs_dim = 64 + self.action_dim
        self.reference = reference if reference is not None else ensure_static_reference(motion)
        self.motion = self.reference.catalog
        self.default_q = default_q
        self.cfg = cfg
        self.recovery_enabled = bool(recovery_enabled)
        midpoint = (kin.q_lo + kin.q_hi) * 0.5
        half_range = (kin.q_hi - kin.q_lo) * 0.5
        self.soft_lo = midpoint - half_range * cfg.soft_limit_factor
        self.soft_hi = midpoint + half_range * cfg.soft_limit_factor
        self.obs_noise = torch.cat((OBS_NOISE[:64], torch.zeros(self.action_dim))).to(default_q.device)

    def obs(
        self,
        state: tuple[torch.Tensor, ...],
        a_prev: torch.Tensor,
        xi: torch.Tensor,
    ) -> torch.Tensor:
        """Return proprioception only; reference orientation is intentionally absent."""
        _, rotation, _, angular_velocity, joint_pos, joint_vel = state
        gravity_body = -rotation[..., 2, :]
        angular_velocity_body = (
            rotation.transpose(-1, -2) @ angular_velocity.unsqueeze(-1)
        ).squeeze(-1)
        observation = torch.cat(
            (
                joint_pos - self.default_q - xi[..., 1:30],
                joint_vel,
                angular_velocity_body,
                gravity_body,
                a_prev,
            ),
            dim=-1,
        )
        if observation.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"Internal observation width {observation.shape[-1]} != {self.obs_dim}."
            )
        return observation

    def add_noise(self, observation: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Add Table A.1 uniform sensor noise to a noise-free observation."""
        draw = (torch.rand_like(observation) if generator is None else
                torch.rand(observation.shape, device=observation.device, dtype=observation.dtype, generator=generator))
        return observation + (
            2.0 * draw - 1.0
        ) * self.obs_noise

    def reward(
        self,
        state: tuple[torch.Tensor, ...],
        reference: ReferenceBatch,
        action: torch.Tensor,
        a_prev: torch.Tensor,
        terms: bool = False,
        *,
        body_state: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the Table A.2 tracking reward multiplied by the 20 ms step."""
        cfg = self.cfg
        ref = reference.now.detach()
        p, rotation, velocity, angular_velocity, joint_pos, joint_vel = state
        body_p, body_R, body_v, body_w = body_state if body_state is not None else self.kin.forward(
            p, rotation, velocity, angular_velocity, joint_pos, joint_vel
        )
        reference_p = ref.p
        reference_R = ref.R

        # BeyondMimic re-anchors reference bodies to robot XY and heading.
        anchor_rotation = self._yaw(rotation @ reference_R.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], reference_p[..., 2:3]), dim=-1)
        anchored_p = anchor.unsqueeze(-2) + (
            anchor_rotation.unsqueeze(-3)
            @ (ref.body_p - reference_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        anchored_R = anchor_rotation.unsqueeze(-3) @ ref.body_R

        body_position_error = ((anchored_p - body_p).square()).sum(-1).mean(-1)
        body_orientation_error = rot_angle_sq(anchored_R, body_R).mean(-1)
        body_linear_velocity_error = (
            (ref.body_v - body_v).square()
        ).sum(-1).mean(-1)
        body_angular_velocity_error = (
            (ref.body_w - body_w).square()
        ).sum(-1).mean(-1)
        root_position_error = (reference_p - p).square().sum(-1)
        root_orientation_error = rot_angle_sq(reference_R, rotation)

        tracking = (
            torch.exp(-body_position_error / cfg.body_pos_std**2)
            + torch.exp(-body_orientation_error / cfg.body_ori_std**2)
            + torch.exp(-body_linear_velocity_error / cfg.body_linvel_std**2)
            + torch.exp(-body_angular_velocity_error / cfg.body_angvel_std**2)
            + 0.5 * torch.exp(-root_position_error / cfg.root_pos_std**2)
            + 0.5 * torch.exp(-root_orientation_error / cfg.root_ori_std**2)
        )
        action_rate = (action - a_prev).square().sum(-1)
        joint_limit_error = (
            torch.relu(joint_pos - self.soft_hi) + torch.relu(self.soft_lo - joint_pos)
        ).sum(-1)
        reward = (
            tracking
            + cfg.w_action_rate * action_rate
            + cfg.w_joint_limit * joint_limit_error
        ) * CONTROL_DT
        if not terms:
            return reward
        return reward, {
            "track": tracking * CONTROL_DT,
            "e_body_pos": body_position_error,
            "e_body_ori": body_orientation_error,
            "e_body_linvel": body_linear_velocity_error,
            "e_body_angvel": body_angular_velocity_error,
            "e_root_pos": root_position_error,
            "e_root_ori": root_orientation_error,
            "action_rate": action_rate,
            "joint_limit": joint_limit_error,
        }

    def termination(
        self,
        state: tuple[torch.Tensor, ...],
        reference: ReferenceBatch,
        *,
        body_state: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Return raw hard tracking failures; only the real manager samples them."""
        if self.recovery_enabled:
            from .recovery import ENDPOINT_THRESHOLD

            errors = self.tracking_errors(state, reference, body_state=body_state)
            raw = torch.cat((errors[..., :1] > self.cfg.term_height,
                             errors[..., 1:2] > self.cfg.term_tilt,
                             errors[..., 2:] > ENDPOINT_THRESHOLD), -1)
            return raw.any(-1)
        p, rotation = state[0], state[1]
        reference_p = reference.now.p.detach()
        reference_R = reference.now.R.detach()
        height_error = (reference_p[..., 2] - p[..., 2]).abs()
        tilt_error = (reference_R[..., 2, 2] - rotation[..., 2, 2]).abs()
        return (height_error > self.cfg.term_height) | (
            tilt_error > self.cfg.term_tilt
        )

    def soft_termination(
        self,
        state: tuple[torch.Tensor, ...],
        reference: ReferenceBatch,
        tz: float = 0.02,
        tt: float = 0.04,
        *,
        recovery_role: torch.Tensor | None = None,
        ep_step: torch.Tensor | None = None,
        body_state: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Return differentiable termination probability for imagined rollouts."""
        if self.recovery_enabled:
            from .recovery import soft_tracking_hazard

            if recovery_role is None or ep_step is None:
                raise ValueError("Recovery termination requires recovery_role and pre-step ep_step")
            return soft_tracking_hazard(
                self.tracking_errors(state, reference, body_state=body_state), recovery_role, ep_step,
                term_height=self.cfg.term_height, term_tilt=self.cfg.term_tilt,
            )
        p, rotation = state[0], state[1]
        reference_p = reference.now.p.detach()
        reference_R = reference.now.R.detach()
        height_error = (reference_p[..., 2] - p[..., 2]).abs()
        tilt_error = (reference_R[..., 2, 2] - rotation[..., 2, 2]).abs()
        alive = (
            1.0
            - torch.sigmoid(
                (height_error - self.cfg.term_height) / tz
            )
        ) * (
            1.0
            - torch.sigmoid((tilt_error - self.cfg.term_tilt) / tt)
        )
        return 1.0 - alive

    def _yaw(self, rotation):
        if self.recovery_enabled:
            from .recovery import recovery_yaw_mat
            return recovery_yaw_mat(rotation)
        return yaw_mat(rotation)

    def tracking_errors(self, state, reference: ReferenceBatch, *, body_state=None):
        from .recovery import tracking_errors

        bodies = self.kin.forward(*state) if body_state is None else body_state
        return tracking_errors(state, reference, bodies)

    def critic_features(
        self,
        state: tuple[torch.Tensor, ...],
        reference: ReferenceBatch,
        xi_n: torch.Tensor,
        *,
        remaining: torch.Tensor | None = None,
        body_state: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Return invariant state, reference errors, and normalized context."""
        ref = reference.now.detach()
        p, rotation, velocity, angular_velocity, joint_pos, joint_vel = state
        heading = self._yaw(rotation)
        heading_transpose = heading.transpose(-1, -2)
        reference_p = ref.p
        reference_R = ref.R
        body_p = body_state[0] if body_state is not None else self.kin.forward(
            p, rotation, velocity, angular_velocity, joint_pos, joint_vel
        )[0]
        anchor_rotation = self._yaw(rotation @ reference_R.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], reference_p[..., 2:3]), dim=-1)
        anchored_p = anchor.unsqueeze(-2) + (
            anchor_rotation.unsqueeze(-3)
            @ (ref.body_p - reference_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        body_error = (
            heading_transpose.unsqueeze(-3)
            @ (anchored_p - body_p).unsqueeze(-1)
        ).squeeze(-1).flatten(-2)
        root_position_error = (
            heading_transpose @ (reference_p - p).unsqueeze(-1)
        ).squeeze(-1)
        orientation_error = relative_root_orientation(rotation, reference_R)
        features = torch.cat(
            (
                invariant_features(state, heading=heading) if self.recovery_enabled else invariant_features(state),
                ref.q - joint_pos,
                body_error,
                root_position_error,
                orientation_error,
                xi_n,
            ),
            dim=-1,
        )
        if features.shape[-1] != INV_DIM + CRITIC_EXTRA:
            raise RuntimeError(
                f"Internal critic width {features.shape[-1]} != "
                f"{INV_DIM + CRITIC_EXTRA}."
            )
        if self.recovery_enabled:
            if remaining is None or remaining.shape != reference.now.p.shape[:-1]:
                raise ValueError("Recovery critic requires one normalized remaining scalar per state")
            features = torch.cat((features, remaining.unsqueeze(-1)), -1)
        return features


__all__ = [
    "ACT_DIM",
    "CRITIC_EXTRA",
    "INV_DIM",
    "MAX_STEP_REWARD",
    "MDP",
    "NUM_J",
    "OBS_DIM",
    "OBS_NOISE",
    "RewardCfg",
    "STATE_DIM",
    "XI_DIM",
    "flatten_state",
    "invariant_features",
    "normalize_xi",
    "relative_root_orientation",
    "RELATIVE_ORIENTATION_DIM",
    "RELATIVE_ORIENTATION_NOISE",
    "unflatten_state",
]
