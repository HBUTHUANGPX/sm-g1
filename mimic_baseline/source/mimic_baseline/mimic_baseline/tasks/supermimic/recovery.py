"""Coupled large-pose RSI and tracking hazards; all runtime geometry is on GPU."""

from __future__ import annotations

import torch

from .kinematics import RXP2_TRACKED_BODY_NAMES
from .motion import CONTROL_DT


RECOVERY_PROBABILITY = 0.15
GROUND_CLEARANCE = 0.15
PROTECTION_SECONDS = 4.0
PROTECTION_STEPS = round(PROTECTION_SECONDS / CONTROL_DT)
ENDPOINT_THRESHOLD = 0.20
POSITION_WIDTH = 0.02
TILT_WIDTH = 0.04
ROOT_TIME_CONSTANT = 3.0
ENDPOINT_TIME_CONSTANT = 0.5
TRACKING_GROUPS = ("height", "tilt", "ankle", "wrist")
_WAIST = RXP2_TRACKED_BODY_NAMES.index("waist_pitch_link")
_ANKLES = tuple(RXP2_TRACKED_BODY_NAMES.index(name) for name in ("l_ankle_roll_link", "r_ankle_roll_link"))
_WRISTS = tuple(RXP2_TRACKED_BODY_NAMES.index(name) for name in ("l_wrist_pitch_link", "r_wrist_pitch_link"))


def remaining_protection(recovery_role: torch.Tensor, ep_step: torch.Tensor) -> torch.Tensor:
    """Normalized remaining time at the source state, never a new replay clock."""
    return recovery_role * ((PROTECTION_STEPS - ep_step).clamp(0, PROTECTION_STEPS).float() / PROTECTION_STEPS)


def _probabilities(dt: float) -> tuple[float, float]:
    # Kernel scalar arguments avoid constructing/copying tiny host tensors.
    return min(max(dt / ROOT_TIME_CONSTANT, 0.), 1.), min(max(dt / ENDPOINT_TIME_CONSTANT, 0.), 1.)


def recovery_provenance(enabled: bool, term_height=.20, term_tilt=.8, *, action_dim: int = 27) -> dict:
    """Describe the actual fixed scheme in checkpoints and the run config."""

    root_probability, endpoint_probability = _probabilities(CONTROL_DT)
    parameters = (
        (term_height, ROOT_TIME_CONSTANT, root_probability, POSITION_WIDTH),
        (term_tilt, ROOT_TIME_CONSTANT, root_probability, TILT_WIDTH),
        (ENDPOINT_THRESHOLD, ENDPOINT_TIME_CONSTANT, endpoint_probability, POSITION_WIDTH),
        (ENDPOINT_THRESHOLD, ENDPOINT_TIME_CONSTANT, endpoint_probability, POSITION_WIDTH),
    )
    return {
        "enabled": bool(enabled),
        "reset_probability": RECOVERY_PROBABILITY,
        "protection_seconds": PROTECTION_SECONDS,
        "protection_steps": PROTECTION_STEPS,
        "ground_clearance_m": GROUND_CLEARANCE,
        "control_dt": CONTROL_DT,
        "tracking_groups": {
            name: {"threshold": threshold, "time_constant_s": seconds,
                   "probability_per_step": probability, "smoothing_width": width}
            for name, (threshold, seconds, probability, width) in zip(TRACKING_GROUPS, parameters, strict=True)
        },
        "real_hazard": "independent_group_bernoulli_or",
        "imagined_hazard": "gate * (1 - product(1 - probability * sigmoid_score))",
        "protection_clock": "pre_step_episode_age",
        "tracking_state": "post_step_state_and_next_reference",
        "critic_context": "role * clamp(200 - episode_age, 0, 200) / 200",
        "critic_extra_scalars": int(enabled),
        "actor_observation_dim": 64 + action_dim,
        "action_dim": action_dim,
        "actor_unchanged": True,
        "time_limit_semantics": "real_collection_truncation_only",
    }


def sample_tracking_terminations(raw, recovery_role, ep_step, *, dt=CONTROL_DT):
    """Four independent group draws, gated by the pre-transition age."""
    draws = torch.rand(raw.shape, device=raw.device)
    root_probability, endpoint_probability = _probabilities(dt)
    accepted = torch.cat((draws[..., :2] < root_probability, draws[..., 2:] < endpoint_probability), -1)
    gate = ~(recovery_role & (ep_step < PROTECTION_STEPS))
    return raw & accepted & gate[..., None]


def soft_tracking_hazard(errors, recovery_role, ep_step, *, term_height=.20, term_tilt=.8, dt=CONTROL_DT):
    """Expected group-OR hazard, with one probability factor per group."""
    root_probability, endpoint_probability = _probabilities(dt)
    height_alive = 1 - root_probability * torch.sigmoid((errors[..., 0] - term_height) / POSITION_WIDTH)
    tilt_alive = 1 - root_probability * torch.sigmoid((errors[..., 1] - term_tilt) / TILT_WIDTH)
    endpoints_alive = 1 - endpoint_probability * torch.sigmoid((errors[..., 2:] - ENDPOINT_THRESHOLD) / POSITION_WIDTH)
    hazard = 1 - height_alive * tilt_alive * endpoints_alive.prod(-1)
    gate = ~(recovery_role & (ep_step < PROTECTION_STEPS))
    # Multiplication deliberately preserves NaN even on protected rows.
    return gate * hazard


def recovery_yaw_mat(matrix: torch.Tensor) -> torch.Tensor:
    """Usual yaw, with a finite transverse-axis choice at a vertical X axis.

    Heading is undefined there. Select the projected Y axis before atan2 so
    even a masked row never differentiates atan2(0, 0). Nonfinite inputs remain
    nonfinite; this only resolves a finite rotation's coordinate singularity.
    """
    x, y = matrix[..., 0, 0], matrix[..., 1, 0]
    vertical = x.square() + y.square() < 1e-12
    x = torch.where(vertical, matrix[..., 1, 1], x)
    y = torch.where(vertical, -matrix[..., 0, 1], y)
    angle = torch.atan2(y, x)
    c, s = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(matrix.shape)


def tracking_errors(state, reference, body_state):
    """Base height/tilt plus max ankle/wrist distance in RGMT waist alignment."""
    p, rotation = state[:2]
    body_p, body_r = body_state[:2]
    ref = reference.now.detach()
    reference_p, reference_r = ref.body_p, ref.body_R
    ref_anchor = reference_p[..., _WAIST, :]
    anchor = torch.cat((body_p[..., _WAIST, :2], ref_anchor[..., 2:3]), -1)
    heading = recovery_yaw_mat(body_r[..., _WAIST, :, :] @ reference_r[..., _WAIST, :, :].transpose(-1, -2))
    aligned = anchor.unsqueeze(-2) + (
        heading.unsqueeze(-3) @ (reference_p - ref_anchor.unsqueeze(-2)).unsqueeze(-1)
    ).squeeze(-1)
    distances = torch.linalg.vector_norm(aligned - body_p, dim=-1)
    return torch.stack((
        (ref.p[..., 2] - p[..., 2]).abs(),
        (ref.R[..., 2, 2] - rotation[..., 2, 2]).abs(),
        torch.maximum(distances[..., _ANKLES[0]], distances[..., _ANKLES[1]]),
        torch.maximum(distances[..., _WRISTS[0]], distances[..., _WRISTS[1]]),
    ), -1)


class RecoveryPlacement:
    """Load headed collision boxes once; place the actual perturbed FK pose."""

    def __init__(self, kin, usd_path):
        from mimic_baseline.utils.collision_bounds import read_collision_bounds

        bounds = read_collision_bounds(usd_path)
        if not bounds or "neck_pitch_link" not in bounds:
            raise ValueError("Recovery requires headed RX-P2 collision bounds, including neck_pitch_link")
        self.kin = kin
        self.body_names = tuple(bounds)
        self.body_ids = torch.tensor([kin.body_names.index(name) for name in self.body_names], device=kin.device)
        boxes = torch.tensor(list(bounds.values()), device=kin.device, dtype=torch.float32)
        bits = torch.cartesian_prod(*[torch.arange(2, device=kin.device)] * 3)
        self.local_corners = torch.stack([boxes[:, bits[:, axis], axis] for axis in range(3)], -1)

    def corners(self, p, rotation, q):
        positions, rotations = self.kin.forward_poses(p, rotation, q)
        return positions[..., self.body_ids, None, :] + (
            rotations[..., self.body_ids, None, :, :] @ self.local_corners[..., None]
        ).squeeze(-1)

    def place(self, p, rotation, q, *, obstacles=None, floor_z=None):
        from mimic_baseline.utils.collision_bounds import terrain_clearance_lift

        corners = self.corners(p, rotation, q)
        floor = p.new_zeros(p.shape[:-1]) if floor_z is None else floor_z
        # The caller applies this candidate only to recovery rows. Its lowest
        # collision corner starts 15 cm above the floor, then clears solids.
        lift = floor + GROUND_CLEARANCE - corners[..., 2].amin(dim=(-1, -2))
        shift = torch.stack((torch.zeros_like(lift), torch.zeros_like(lift), lift), -1)
        placed = p + shift
        if obstacles is not None:
            placed = placed.clone()
            placed[..., 2] += terrain_clearance_lift(
                corners + shift[:, None, None], obstacles, floor, GROUND_CLEARANCE,
            )
        return placed
