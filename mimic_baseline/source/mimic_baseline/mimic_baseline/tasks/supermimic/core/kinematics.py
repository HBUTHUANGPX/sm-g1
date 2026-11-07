"""Differentiable RX-P2 forward kinematics parsed directly from URDF."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from mimic_baseline.tasks.supermimic.core.rotations import axis_angle_to_mat, quat_to_mat


RXP2_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint",
    "r_elbow_joint",
    "r_wrist_roll_joint",
    "r_wrist_pitch_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint",
    "l_elbow_joint",
    "l_wrist_roll_joint",
    "l_wrist_pitch_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_hip_yaw_joint",
    "r_knee_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_hip_yaw_joint",
    "l_knee_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
)

RXP2_TRACKED_BODY_NAMES = (
    "base_link",
    "l_hip_roll_link",
    "l_knee_link",
    "l_ankle_roll_link",
    "r_hip_roll_link",
    "r_knee_link",
    "r_ankle_roll_link",
    "waist_pitch_link",
    "l_shoulder_roll_link",
    "l_elbow_link",
    "l_wrist_pitch_link",
    "r_shoulder_roll_link",
    "r_elbow_link",
    "r_wrist_pitch_link",
)


def _parse_vector(value: str | None, default: tuple[float, float, float]) -> tuple[float, ...]:
    if value is None:
        return default
    parts = tuple(float(part) for part in value.split())
    if len(parts) != len(default):
        raise ValueError(f"Expected {len(default)} values, got {value!r}.")
    return parts


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> torch.Tensor:
    """URDF fixed-axis roll-pitch-yaw, ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=torch.float32,
    )


class RXP2Kinematics:
    """Analytic RX-P2 link-origin FK for the canonical 27 body joints plus neck."""

    def __init__(self, urdf_path: str | Path, device: str | torch.device, *,
                 joint_names=RXP2_JOINT_NAMES, tracked_body_names=RXP2_TRACKED_BODY_NAMES,
                 root_body_name="base_link", action_dim=27):
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        self.device = torch.device(device)
        self.joint_names = tuple(joint_names)
        self.tracked_body_names = tuple(tracked_body_names)
        self.root_body_name = root_body_name
        self.action_dim = action_dim
        if not 0 < action_dim <= len(self.joint_names):
            raise ValueError("action_dim must select a nonempty prefix of canonical joints")
        self._parse_urdf()

    def _parse_urdf(self) -> None:
        root = ET.parse(self.urdf_path).getroot()
        link_names = [element.attrib["name"] for element in root.findall("link")]
        if len(link_names) != len(set(link_names)):
            raise ValueError("URDF contains duplicate link names.")
        link_index = {name: index for index, name in enumerate(link_names)}
        if self.root_body_name not in link_index:
            raise ValueError(f"URDF is missing root link {self.root_body_name!r}.")

        joint_by_name = {}
        child_link = {}
        for element in root.findall("joint"):
            joint_type = element.attrib["type"]
            if joint_type not in {"revolute", "continuous", "fixed"}:
                raise ValueError(
                    f"Unsupported RX-P2 URDF joint type {joint_type!r} for "
                    f"{element.attrib['name']!r}."
                )
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError(f"URDF joint {element.attrib['name']!r} lacks parent or child.")
            parent_name = parent.attrib["link"]
            child_name = child.attrib["link"]
            if child_name in child_link:
                raise ValueError(f"URDF link {child_name!r} has multiple parent joints.")
            child_link[child_name] = element.attrib["name"]
            joint_by_name[element.attrib["name"]] = element

        children: dict[str, list[str]] = {name: [] for name in link_names}
        for child_name, joint_name in child_link.items():
            parent = joint_by_name[joint_name].find("parent").attrib["link"]
            children[parent].append(child_name)
        ordered_links = [self.root_body_name]
        for link_name in ordered_links:
            ordered_links.extend(children[link_name])
        if len(ordered_links) != len(link_names):
            missing = sorted(set(link_names) - set(ordered_links))
            raise ValueError(f"URDF links are not connected to base_link: {missing}.")

        parents: list[int] = []
        offsets_p: list[tuple[float, float, float]] = []
        offsets_r: list[torch.Tensor] = []
        joint_indices: list[int] = []
        axes: list[tuple[float, float, float]] = []
        canonical_joint_index = {
            name: index for index, name in enumerate(self.joint_names)
        }
        present_joints: set[str] = set()
        lower = torch.full((len(self.joint_names),), -math.inf)
        upper = torch.full((len(self.joint_names),), math.inf)

        for link_name in ordered_links:
            if link_name == self.root_body_name:
                parents.append(-1)
                offsets_p.append((0.0, 0.0, 0.0))
                offsets_r.append(torch.eye(3, dtype=torch.float32))
                joint_indices.append(-1)
                axes.append((0.0, 0.0, 0.0))
                continue
            joint = joint_by_name[child_link[link_name]]
            parent_name = joint.find("parent").attrib["link"]
            parents.append(ordered_links.index(parent_name))
            origin = joint.find("origin")
            offsets_p.append(
                _parse_vector(
                    None if origin is None else origin.attrib.get("xyz"),
                    (0.0, 0.0, 0.0),
                )
            )
            offsets_r.append(
                _rpy_to_matrix(
                    _parse_vector(
                        None if origin is None else origin.attrib.get("rpy"),
                        (0.0, 0.0, 0.0),
                    )
                )
            )
            joint_name = joint.attrib["name"]
            if joint.attrib["type"] == "fixed":
                joint_indices.append(-1)
                axes.append((0.0, 0.0, 0.0))
                continue
            if joint_name not in canonical_joint_index:
                raise ValueError(f"URDF actuated joint {joint_name!r} is not canonical.")
            index = canonical_joint_index[joint_name]
            if joint_name in present_joints:
                raise ValueError(f"Canonical joint {joint_name!r} appears more than once.")
            present_joints.add(joint_name)
            joint_indices.append(index)
            axis = joint.find("axis")
            axes.append(
                _parse_vector(
                    None if axis is None else axis.attrib.get("xyz"),
                    (1.0, 0.0, 0.0),
                )
            )
            limit = joint.find("limit")
            if limit is not None and "lower" in limit.attrib:
                lower[index] = float(limit.attrib["lower"])
            if limit is not None and "upper" in limit.attrib:
                upper[index] = float(limit.attrib["upper"])

        missing_joints = sorted(set(self.joint_names) - present_joints)
        if missing_joints:
            raise ValueError(f"RX-P2 URDF is missing canonical joints: {missing_joints}.")
        missing_tracked = sorted(set(self.tracked_body_names) - set(link_names))
        if missing_tracked:
            raise ValueError(f"RX-P2 URDF is missing tracked links: {missing_tracked}.")

        self.body_names = tuple(ordered_links)
        self.parent = tuple(parents)
        self.joint_idx = tuple(joint_indices)
        self.q_lo = lower.to(self.device)
        self.q_hi = upper.to(self.device)
        self.off_p = torch.tensor(offsets_p, dtype=torch.float32, device=self.device)
        self.off_R = torch.stack(offsets_r).to(self.device)
        self.axes = torch.tensor(axes, dtype=torch.float32, device=self.device)
        self._pose_joint_indices = torch.tensor([max(index, 0) for index in joint_indices], device=self.device)
        self._pose_moving = torch.tensor([index >= 0 for index in joint_indices], device=self.device)
        self.tracked_idx = tuple(
            ordered_links.index(name) for name in self.tracked_body_names
        )
        if not torch.isfinite(self.q_lo).all() or not torch.isfinite(self.q_hi).all():
            raise ValueError("Every canonical RX-P2 joint must declare finite limits.")

    def forward_poses(self, p: torch.Tensor, R: torch.Tensor, q: torch.Tensor):
        """Return all link-origin positions/rotations in ``body_names`` order.

        The pose-only path includes head and self-occlusion links, using the
        same URDF transforms as forward without computing unused velocities.
        """
        if q.shape[-1] != len(self.joint_names):
            raise ValueError(f"Expected q ending in {len(self.joint_names)}, got {tuple(q.shape)}.")
        angles = torch.where(self._pose_moving, q[..., self._pose_joint_indices], 0.)
        joint_rotations = axis_angle_to_mat(self.axes, angles)
        positions, rotations = [], []
        for link_index, parent_index in enumerate(self.parent):
            if parent_index < 0:
                positions.append(p)
                rotations.append(R)
                continue
            parent_rotation = rotations[parent_index]
            positions.append(positions[parent_index] + (
                parent_rotation @ self.off_p[link_index, :, None]
            ).squeeze(-1))
            rotation = parent_rotation @ self.off_R[link_index]
            joint_index = self.joint_idx[link_index]
            if joint_index >= 0:
                rotation = rotation @ joint_rotations[..., link_index, :, :]
            rotations.append(rotation)
        return torch.stack(positions, dim=-2), torch.stack(rotations, dim=-3)

    def forward(
        self,
        p: torch.Tensor,
        R: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tracked link-origin pose and world velocity tensors.

        Each output has shape ``(..., 14, ...)`` and velocities are computed by
        analytic rigid-body transport plus the joint-axis angular-velocity term.
        """
        if q.shape[-1] != len(self.joint_names) or qd.shape != q.shape:
            raise ValueError(
                f"Expected q and qd ending in {len(self.joint_names)}, got "
                f"{tuple(q.shape)} and {tuple(qd.shape)}."
            )
        positions: list[torch.Tensor | None] = [None] * len(self.body_names)
        rotations: list[torch.Tensor | None] = [None] * len(self.body_names)
        velocities: list[torch.Tensor | None] = [None] * len(self.body_names)
        angular_velocities: list[torch.Tensor | None] = [None] * len(self.body_names)
        for link_index in range(len(self.body_names)):
            parent_index = self.parent[link_index]
            if parent_index < 0:
                positions[link_index] = p
                rotations[link_index] = R
                velocities[link_index] = v
                angular_velocities[link_index] = w
                continue
            parent_position = positions[parent_index]
            parent_rotation = rotations[parent_index]
            parent_velocity = velocities[parent_index]
            parent_angular_velocity = angular_velocities[parent_index]
            link_position = parent_position + (
                parent_rotation @ self.off_p[link_index, :, None]
            ).squeeze(-1)
            fixed_rotation = parent_rotation @ self.off_R[link_index]
            joint_index = self.joint_idx[link_index]
            if joint_index < 0:
                link_rotation = fixed_rotation
                link_angular_velocity = parent_angular_velocity
            else:
                link_rotation = fixed_rotation @ axis_angle_to_mat(
                    self.axes[link_index], q[..., joint_index]
                )
                world_axis = (fixed_rotation @ self.axes[link_index, :, None]).squeeze(-1)
                link_angular_velocity = (
                    parent_angular_velocity
                    + world_axis * qd[..., joint_index : joint_index + 1]
                )
            link_velocity = parent_velocity + torch.cross(
                parent_angular_velocity,
                link_position - parent_position,
                dim=-1,
            )
            positions[link_index] = link_position
            rotations[link_index] = link_rotation
            velocities[link_index] = link_velocity
            angular_velocities[link_index] = link_angular_velocity

        indices = self.tracked_idx
        return (
            torch.stack([positions[index] for index in indices], dim=-2),
            torch.stack([rotations[index] for index in indices], dim=-3),
            torch.stack([velocities[index] for index in indices], dim=-2),
            torch.stack([angular_velocities[index] for index in indices], dim=-2),
        )


__all__ = [
    "RXP2_JOINT_NAMES",
    "RXP2_TRACKED_BODY_NAMES",
    "RXP2Kinematics",
]
