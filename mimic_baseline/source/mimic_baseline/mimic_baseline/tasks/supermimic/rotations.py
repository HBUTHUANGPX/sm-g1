"""Batched rotations used by the SuperMimic state and kinematics interfaces.

Quaternions use scalar-first ``wxyz`` order. Rotation matrices act on column
vectors. The matrix-to-quaternion branch selection follows the standard
Shoemake construction so the selected square-root is away from zero near a
half turn.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``wxyz`` quaternions to rotation matrices."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to ``wxyz`` quaternions.

    Each matrix element is used by only one of four candidate formulas. The
    candidate with the largest base component is selected, which avoids
    dividing by values close to zero at identity and half-turn rotations.
    """
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    trace = m00 + m11 + m22
    branch = torch.stack((trace, m00, m11, m22), dim=-1).argmax(dim=-1)

    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + trace, min=1e-12))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-12))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-12))

    candidates = torch.stack(
        (
            torch.stack(
                (qw, (m21 - m12) / (4.0 * qw), (m02 - m20) / (4.0 * qw), (m10 - m01) / (4.0 * qw)),
                dim=-1,
            ),
            torch.stack(
                ((m21 - m12) / (4.0 * qx), qx, (m01 + m10) / (4.0 * qx), (m02 + m20) / (4.0 * qx)),
                dim=-1,
            ),
            torch.stack(
                ((m02 - m20) / (4.0 * qy), (m01 + m10) / (4.0 * qy), qy, (m12 + m21) / (4.0 * qy)),
                dim=-1,
            ),
            torch.stack(
                ((m10 - m01) / (4.0 * qz), (m02 + m20) / (4.0 * qz), (m12 + m21) / (4.0 * qz), qz),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    quat = torch.gather(
        candidates,
        -2,
        branch[..., None, None].expand(branch.shape + (1, 4)),
    ).squeeze(-2)
    quat = F.normalize(quat, dim=-1)
    return quat * torch.where(quat[..., :1] < 0.0, -1.0, 1.0)


def skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric cross-product matrix for ``(..., 3)`` vectors."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula for fixed unit axes and arbitrary leading dimensions."""
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = axis.unbind(-1)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return torch.stack(
        (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Exponential map from rotation vectors, with a finite Taylor branch at zero."""
    theta_sq = (rotation_vector * rotation_vector).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq + 1e-12)
    small = theta_sq < 1e-8
    coefficient_a = torch.where(
        small,
        1.0 - theta_sq / 6.0,
        torch.sin(theta) / theta,
    )
    coefficient_b = torch.where(
        small,
        0.5 - theta_sq / 24.0,
        (1.0 - torch.cos(theta)) / (theta_sq + 1e-12),
    )
    generator = skew(rotation_vector)
    identity = torch.eye(
        3, dtype=rotation_vector.dtype, device=rotation_vector.device
    ).expand(generator.shape)
    return (
        identity
        + coefficient_a[..., None] * generator
        + coefficient_b[..., None] * (generator @ generator)
    )


def quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to principal rotation vectors."""
    quat = F.normalize(quat, dim=-1)
    quat = quat * torch.where(quat[..., :1] < 0.0, -1.0, 1.0)
    vector = quat[..., 1:]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    small = sin_half < 1e-6
    scale = torch.where(
        small,
        2.0 / quat[..., :1].clamp_min(1e-6),
        2.0
        * torch.atan2(sin_half, quat[..., :1])
        / (sin_half + 1e-12),
    )
    return vector * scale


def so3_log(matrix: torch.Tensor) -> torch.Tensor:
    """Principal logarithmic map of rotation matrices."""
    return quat_to_rotvec(mat_to_quat(matrix))


def rot_angle_sq(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between two rotations.

    The small-angle branch is a polynomial in the relative skew vector, avoiding
    the non-differentiable normalization at identity. It is selected from the
    trace, not from ``sine_sq``, because the sine also vanishes at a half turn.
    The general branch uses ``atan2`` and remains finite and accurate at pi.
    """
    trace = (first * second).sum(dim=(-1, -2))
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    relative = first.transpose(-1, -2) @ second
    sine_sq = (
        (relative[..., 2, 1] - relative[..., 1, 2]).square()
        + (relative[..., 0, 2] - relative[..., 2, 0]).square()
        + (relative[..., 1, 0] - relative[..., 0, 1]).square()
    ) * 0.25
    small = cosine > 1.0 - 1e-8
    small_angle_sq = sine_sq * (1.0 + sine_sq / 6.0)
    sine_sq_safe = sine_sq.clamp_min(torch.finfo(sine_sq.dtype).tiny)
    general_angle_sq = torch.atan2(torch.sqrt(sine_sq_safe), cosine).square()
    return torch.where(small, small_angle_sq, general_angle_sq)


def yaw_mat(matrix: torch.Tensor) -> torch.Tensor:
    """Extract heading as a yaw-only rotation matrix."""
    yaw = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack(
        (cosine, -sine, zero, sine, cosine, zero, zero, zero, one),
        dim=-1,
    ).reshape(matrix.shape[:-2] + (3, 3))


def mat_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Return the first two matrix columns as a continuous 6-D rotation."""
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def rot6d_to_mat(vector: torch.Tensor) -> torch.Tensor:
    """Project a 6-D rotation onto SO(3) with Gram-Schmidt.

    The inverse of :func:`mat_to_6d`. The input is not assumed orthonormal, so
    the output is a valid rotation matrix even when the world model emits a raw
    regression output. Gradients flow through both columns.
    """
    if vector.shape[-1] != 6:
        raise ValueError(f"Expected 6-D rotation, got last dim {vector.shape[-1]}.")
    # mat_to_6d flattens the first two columns row-major:
    # [R00, R01, R10, R11, R20, R21], so the columns interleave every 2 entries.
    columns = vector.reshape(vector.shape[:-1] + (3, 2))
    first = columns[..., 0]
    second = columns[..., 1]
    first = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    second = second - (first * second).sum(-1, keepdim=True) * first
    second = second / second.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def quat_slerp(
    start: torch.Tensor,
    end: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    """Shortest-arc spherical interpolation for ``wxyz`` quaternions."""
    dot = (start * end).sum(dim=-1, keepdim=True)
    end = torch.where(dot < 0.0, -end, end)
    dot = dot.abs().clamp(max=1.0)
    angle = torch.acos(dot)
    sine = torch.sin(angle)
    fraction = fraction[..., None]
    start_weight = torch.where(
        sine < 1e-6,
        1.0 - fraction,
        torch.sin((1.0 - fraction) * angle) / (sine + 1e-12),
    )
    end_weight = torch.where(
        sine < 1e-6,
        fraction,
        torch.sin(fraction * angle) / (sine + 1e-12),
    )
    return F.normalize(start_weight * start + end_weight * end, dim=-1)


__all__ = [
    "axis_angle_to_mat",
    "mat_to_6d",
    "mat_to_quat",
    "quat_slerp",
    "quat_to_mat",
    "quat_to_rotvec",
    "rot_angle_sq",
    "skew",
    "so3_exp",
    "so3_log",
    "yaw_mat",
]
