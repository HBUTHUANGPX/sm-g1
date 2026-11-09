"""G1 metadata for the original ManagerBased SuperMimic implementation.

Joint order is the LAFAN1 retarget CSV order. FK uses the original project's
URDF parser and differentiable tree recursion, including fixed links.
"""

import torch

from mimic_baseline.tasks.supermimic.core.kinematics import RXP2Kinematics
from mimic_baseline.tasks.supermimic.core.rotations import axis_angle_to_mat

G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


class G1Kinematics(RXP2Kinematics):
    """G1 configuration of the existing URDF FK, with all 29 joints actuated."""

    recovery_required_bodies = ("pelvis_contour_link", "torso_link")

    def __init__(self, urdf_path, device):
        super().__init__(
            urdf_path,
            device,
            joint_names=G1_JOINT_NAMES,
            tracked_body_names=G1_TRACKED_BODY_NAMES,
            root_body_name="pelvis",
            action_dim=29,
        )

        index = self.tracked_body_names.index
        self.recovery_body_indices = (
            index("torso_link"),
            (index("left_ankle_roll_link"), index("right_ankle_roll_link")),
            (index("left_wrist_yaw_link"), index("right_wrist_yaw_link")),
        )

    def _initialize_fast_topology(self):
        # The parser orders links breadth-first. Cache only integer topology,
        # retaining the original tensors as the single source of transforms.
        levels = [[0]]
        while True:
            previous = {index: slot for slot, index in enumerate(levels[-1])}
            children = [i for i, parent in enumerate(self.parent) if parent in previous]
            if not children:
                break
            levels.append(children)
        order = [index for level in levels for index in level]
        if order != list(range(len(self.body_names))):
            raise ValueError("Fast G1 FK requires breadth-first body ordering.")
        self._fast_levels = tuple(
            (
                torch.tensor(level, device=self.device),
                torch.tensor([levels[depth - 1].index(self.parent[i]) for i in level],
                             device=self.device),
            )
            for depth, level in enumerate(levels) if depth
        )
        self._fast_tracked = torch.tensor(self.tracked_idx, device=self.device)

    def forward_fast(self, p, R, v, w, q, qd):
        """Optional depth-batched FK with the reference forward contract.

        This is opt-in: inherited ``forward`` remains the default. No tensors
        are detached or written in-place, preserving BPTT gradients. Fixed
        joints use identity rotations and zero angular velocity increments.
        """
        if q.shape[-1] != len(self.joint_names) or qd.shape != q.shape:
            raise ValueError(
                f"Expected q and qd ending in {len(self.joint_names)}, got "
                f"{tuple(q.shape)} and {tuple(qd.shape)}."
            )
        if not hasattr(self, "_fast_levels"):
            self._initialize_fast_topology()
        angles = torch.where(self._pose_moving, q[..., self._pose_joint_indices], 0.)
        rates = torch.where(self._pose_moving, qd[..., self._pose_joint_indices], 0.)
        joint_rotations = axis_angle_to_mat(self.axes, angles)
        positions, rotations = [p.unsqueeze(-2)], [R.unsqueeze(-3)]
        velocities, angular_velocities = [v.unsqueeze(-2)], [w.unsqueeze(-2)]
        for links, parents in self._fast_levels:
            parent_p = positions[-1].index_select(-2, parents)
            parent_R = rotations[-1].index_select(-3, parents)
            parent_v = velocities[-1].index_select(-2, parents)
            parent_w = angular_velocities[-1].index_select(-2, parents)
            link_p = parent_p + (parent_R @ self.off_p[links, :, None]).squeeze(-1)
            fixed_R = parent_R @ self.off_R[links]
            link_R = fixed_R @ joint_rotations.index_select(-3, links)
            world_axis = (fixed_R @ self.axes[links, :, None]).squeeze(-1)
            link_w = parent_w + world_axis * rates.index_select(-1, links).unsqueeze(-1)
            link_v = parent_v + torch.cross(parent_w, link_p - parent_p, dim=-1)
            positions.append(link_p)
            rotations.append(link_R)
            velocities.append(link_v)
            angular_velocities.append(link_w)
        return tuple(
            torch.cat(values, dim=dim).index_select(dim, self._fast_tracked)
            for values, dim in ((positions, -2), (rotations, -3),
                                (velocities, -2), (angular_velocities, -2))
        )


class G1BatchedKinematics(G1Kinematics):
    """Opt-in batched FK whose method remains correctly bound after copying.

    MotionLib shallow-copies kinematics before casting its URDF transforms to
    double precision. A class method alias follows the copied instance, unlike
    assigning an already-bound method to an instance attribute.
    """

    forward = G1Kinematics.forward_fast
