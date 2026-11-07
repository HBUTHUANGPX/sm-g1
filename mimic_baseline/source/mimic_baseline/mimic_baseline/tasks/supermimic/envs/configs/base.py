"""Original ManagerBased configs/base responsibilities."""

from __future__ import annotations

import math

from typing import Literal

import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.managers import (
    ActionTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

from isaaclab.scene import InteractiveSceneCfg

from isaaclab.utils.configclass import configclass

from isaaclab.utils.noise import UniformNoiseCfg

from mimic_baseline.robots.torque_speed_actuator import DelayedTorqueSpeedPDActuatorCfg

from mimic_baseline.tasks.supermimic.envs.actuators import SuperMimicDelayedPDActuator

from mimic_baseline.tasks.supermimic.envs.simulation import GROUND_PLANE_USD, make_sim

from mimic_baseline.tasks.supermimic.envs.mdp.commands import SuperMimicMotionCommandCfg

from mimic_baseline.tasks.supermimic.core.mdp import ACT_DIM, RELATIVE_ORIENTATION_NOISE

from mimic_baseline.tasks.supermimic.data.motion import CONTROL_DT

from mimic_baseline.tasks.supermimic.learning.perception import (
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
)

from mimic_baseline.tasks.supermimic.robots.rxp2.asset import make_robot_cfg

_MDP = "mimic_baseline.tasks.supermimic.envs.mdp"

PHYSICS_DT = 0.005

DECIMATION = 4

ENV_SPACING = 2.5

MAX_EPISODE_LENGTH_S = 10.0

NECK_PITCH_DOWN_29DEG = math.radians(29.0)


@configclass
class SuperMimicObservationsCfg:
    @configclass
    class ObsGroupCfg(ObservationGroupCfg):
        # The two independently controlled neck joints remain in proprioception.
        joint_pos_residual = ObservationTermCfg(
            func=f"{_MDP}.observations:joint_pos_residual",
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObservationTermCfg(
            func=f"{_MDP}.observations:joint_vel",
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        body_ang_vel = ObservationTermCfg(
            func=f"{_MDP}.observations:body_ang_vel",
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObservationTermCfg(
            func=f"{_MDP}.observations:projected_gravity",
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        last_action = ObservationTermCfg(func=f"{_MDP}.observations:last_action")
        concatenate_terms = True
        enable_corruption = True

    @configclass
    class CommandGroupCfg(ObservationGroupCfg):
        motion_goal = ObservationTermCfg(func=f"{_MDP}.observations:motion_goal")
        concatenate_terms = True
        enable_corruption = False

    @configclass
    class RelativeOrientationCfg(ObservationGroupCfg):
        ref_root_rot6d_in_root = ObservationTermCfg(
            func=f"{_MDP}.observations:ref_root_rot6d_in_root",
            noise=UniformNoiseCfg(
                n_min=-RELATIVE_ORIENTATION_NOISE, n_max=RELATIVE_ORIENTATION_NOISE
            ),
        )
        concatenate_terms = True
        enable_corruption = True

    supermimic_obs = ObsGroupCfg()
    supermimic_command = CommandGroupCfg()
    supermimic_relative_orientation: RelativeOrientationCfg | None = None


@configclass
class SuperMimicActionCfg:
    @configclass
    class JointPositionActionCfg(ActionTermCfg):
        class_type = f"{_MDP}.actions:ResidualJointPositionAction"
        asset_name: str = "robot"
        action_dim: int = ACT_DIM

    joint_pos = JointPositionActionCfg()


@configclass
class SuperMimicCommandCfg:
    motion = SuperMimicMotionCommandCfg(step_dt=CONTROL_DT, gamma=0.99)


@configclass
class SuperMimicRewardCfg:
    """The same unscaled terms as MDP.reward; RewardManager multiplies by dt."""

    body_position = RewardTermCfg(
        func=f"{_MDP}.rewards:body_position_tracking",
        weight=1.0,
        params={"std": 0.3},
    )
    body_orientation = RewardTermCfg(
        func=f"{_MDP}.rewards:body_orientation_tracking",
        weight=1.0,
        params={"std": 0.4},
    )
    body_linvel = RewardTermCfg(
        func=f"{_MDP}.rewards:body_linvel_tracking",
        weight=1.0,
        params={"std": 1.0},
    )
    body_angvel = RewardTermCfg(
        func=f"{_MDP}.rewards:body_angvel_tracking",
        weight=1.0,
        params={"std": 3.14},
    )
    root_position = RewardTermCfg(
        func=f"{_MDP}.rewards:root_position_tracking",
        weight=0.5,
        params={"std": 0.3},
    )
    root_orientation = RewardTermCfg(
        func=f"{_MDP}.rewards:root_orientation_tracking",
        weight=0.5,
        params={"std": 0.4},
    )
    action_rate = RewardTermCfg(func=f"{_MDP}.rewards:action_rate_penalty", weight=-0.1)
    joint_limit = RewardTermCfg(func=f"{_MDP}.rewards:joint_limit_penalty", weight=-2.0)


@configclass
class SuperMimicTerminationCfg:
    height = TerminationTermCfg(
        func=f"{_MDP}.terminations:height_termination",
        params={"threshold": 0.20},
    )
    tilt = TerminationTermCfg(
        func=f"{_MDP}.terminations:tilt_termination",
        params={"threshold": 0.8},
    )
    motion_end = TerminationTermCfg(
        func=f"{_MDP}.terminations:motion_end", time_out=True
    )
    time_out = TerminationTermCfg(func=f"{_MDP}.terminations:time_out", time_out=True)
    ankle: TerminationTermCfg | None = None
    wrist: TerminationTermCfg | None = None


@configclass
class SuperMimicEventCfg:
    # The reset event precedes the command's state write. All pose perturbations
    # belong to that write, so a later write cannot erase an event's noise.
    randomize_physics = EventTermCfg(
        func=f"{_MDP}.events:randomize_physics_parameters",
        mode="reset",
    )


@configclass
class SuperMimicCurriculumCfg:
    # Statistics are updated once per vector step, including ranks with no
    # resets. The reset curriculum reports them; it must not update them again.
    update_bin_returns = CurriculumTermCfg(
        func=f"{_MDP}.curriculum:update_motion_bin_returns"
    )


@configclass
class SuperMimicSceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = ENV_SPACING
    replicate_physics: bool = True
    terrain = None
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            usd_path=GROUND_PLANE_USD,
            physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="max",
            ),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class SuperMimicEnvCfg(ManagerBasedRLEnvCfg):
    sim = make_sim("cuda:0")
    decimation: int = DECIMATION
    scene: SuperMimicSceneCfg = SuperMimicSceneCfg()
    observations: SuperMimicObservationsCfg = SuperMimicObservationsCfg()
    actions: SuperMimicActionCfg = SuperMimicActionCfg()
    commands: SuperMimicCommandCfg = SuperMimicCommandCfg()
    rewards: SuperMimicRewardCfg = SuperMimicRewardCfg()
    terminations: SuperMimicTerminationCfg = SuperMimicTerminationCfg()
    events: SuperMimicEventCfg = SuperMimicEventCfg()
    curriculum: SuperMimicCurriculumCfg = SuperMimicCurriculumCfg()
    episode_length_s: float = MAX_EPISODE_LENGTH_S
    actuator_type: str = "delayed_torque_speed"
    min_delay: int = 0
    max_delay: int = 4
    neck_yaw: float = 0.0
    neck_pitch: float = 0.0
    reset_noise: bool = True
    recovery_enabled: bool = False
    eval_mode: bool = False
    clean_perception: bool = False
    raycast_backend: Literal["warp", "ovrtx"] = "warp"
    height_raycast_backend: Literal["warp", "ovrtx"] | None = None
    height_scan_frame: str = "yaw"
    height_scan_size: tuple[float, float] = HEIGHT_SCAN_SIZE
    height_scan_resolution: float = HEIGHT_SCAN_RESOLUTION
    robot_name: str = "rx_p2"
    action_dim: int = ACT_DIM
    torso_body_name: str = "waist_pitch_link"
    robot_urdf: str | None = None
    ground_usd: str | None = None
    robot_usd: str | None = None
    goal_mode: str = "reference"
    actor_relative_orientation: bool = True
    motion_scene_file: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.robot_name == "g1":
            from mimic_baseline.tasks.supermimic.robots.g1.asset import (
                make_g1_robot_cfg,
            )

            self.scene.robot, self.action_scale = make_g1_robot_cfg(
                self.robot_usd,
                self.robot_urdf,
                actuator_type=self.actuator_type,
                min_delay=self.min_delay,
                max_delay=self.max_delay,
                self_collisions=self.self_collisions,
            )
            self.actions.joint_pos.action_dim = self.action_dim
            if self.ground_usd:
                self.scene.ground.spawn.usd_path = self.ground_usd
            return
        self.scene.robot, self.action_scale = make_robot_cfg(
            actuator_type=self.actuator_type,
            min_delay=self.min_delay,
            max_delay=self.max_delay,
            neck_yaw=self.neck_yaw,
            neck_pitch=self.neck_pitch,
            **({"usd_path": self.robot_usd} if self.robot_usd is not None else {}),
        )
        for actuator_cfg in self.scene.robot.actuators.values():
            if isinstance(actuator_cfg, DelayedTorqueSpeedPDActuatorCfg):
                actuator_cfg.class_type = SuperMimicDelayedPDActuator
