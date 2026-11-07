"""G1 metadata for the original ManagerBased SuperMimic implementation.

Joint order is the LAFAN1 retarget CSV order. FK uses the original project's
URDF parser and differentiable tree recursion, including fixed links.
"""

from .kinematics import RXP2Kinematics

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

    recovery_required_bodies = ("pelvis", "torso_link")

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
