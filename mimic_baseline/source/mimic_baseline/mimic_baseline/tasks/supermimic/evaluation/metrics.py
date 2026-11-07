"""Original evaluation metrics functions."""

from __future__ import annotations


import torch


from mimic_baseline.tasks.supermimic.core.mdp import MAX_STEP_REWARD, unflatten_state


from mimic_baseline.tasks.supermimic.data.reference import ReferenceBatch





def _max_step_reward() -> float:
    return float(MAX_STEP_REWARD)


def normalized_clip_return(
    total_return: float,
    clip_start: int,
    clip_end: int,
    *,
    duration_steps: int | None = None,
) -> float:
    """Normalize by the reward ceiling over the reported simulation duration."""
    steps = (
        max(int(clip_end) - int(clip_start), 1)
        if duration_steps is None
        else max(int(duration_steps), 1)
    )
    return float(total_return) / (steps * _max_step_reward())


def _step_metrics(
    executed: dict[str, torch.Tensor],
    reference: ReferenceBatch,
    terms: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute per-environment squared errors on the simulator device."""
    actual = unflatten_state(executed["s1"])
    root_error = actual[0] - reference.now.p
    relative_rotation = actual[1].transpose(-1, -2) @ reference.now.R
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return {
        "root_xy_rmse": root_error[..., :2].square().sum(-1),
        "root_z_rmse": root_error[..., 2].square(),
        "root_rotation_rmse_rad": torch.acos(cosine).square(),
        "joint_rmse": (actual[4] - reference.now.q).square().mean(-1),
        "body_position_rmse": terms["e_body_pos"],
        "body_orientation_rmse_rad": terms["e_body_ori"],
    }
