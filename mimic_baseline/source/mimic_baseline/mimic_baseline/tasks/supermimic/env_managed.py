"""SuperMimic manager environments and the model-based trainer adapter.

The simulator keeps the Gym five-value step API. A task-local reward manager
captures transitions before IsaacLab resets completed episodes; the adapter
passes those transitions to the unchanged model-based learning loop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg, CurriculumTermCfg, EventTermCfg, ObservationGroupCfg,
    ObservationTermCfg, RewardManager, RewardTermCfg, TerminationManager, TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from mimic_baseline.assets import ASSET_DIR
from mimic_baseline.robots.torque_speed_actuator import DelayedTorqueSpeedPDActuatorCfg
from .action_bounds import clip_action
from .actuators import SuperMimicDelayedPDActuator
from .env import GROUND_PLANE_USD, make_sim
from .kinematics import RXP2Kinematics
from .mdp.commands import SuperMimicMotionCommandCfg, canonical_state
from .mdp_legacy import ACT_DIM, MDP, NUM_J, RELATIVE_ORIENTATION_NOISE, RewardCfg, flatten_state
from .motion import CONTROL_DT, MotionLib
from .perception import HEIGHT_SCAN_RESOLUTION, HEIGHT_SCAN_SIZE
from .randomization import EpisodeRandomization
from .reference import StaticReference
from .robot import make_robot_cfg

_MDP = "mimic_baseline.tasks.supermimic.mdp"
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
            noise=UniformNoiseCfg(n_min=-RELATIVE_ORIENTATION_NOISE, n_max=RELATIVE_ORIENTATION_NOISE),
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
        func=f"{_MDP}.rewards:body_position_tracking", weight=1.0,
        params={"std": 0.3},
    )
    body_orientation = RewardTermCfg(
        func=f"{_MDP}.rewards:body_orientation_tracking", weight=1.0,
        params={"std": 0.4},
    )
    body_linvel = RewardTermCfg(
        func=f"{_MDP}.rewards:body_linvel_tracking", weight=1.0,
        params={"std": 1.0},
    )
    body_angvel = RewardTermCfg(
        func=f"{_MDP}.rewards:body_angvel_tracking", weight=1.0,
        params={"std": 3.14},
    )
    root_position = RewardTermCfg(
        func=f"{_MDP}.rewards:root_position_tracking", weight=0.5,
        params={"std": 0.3},
    )
    root_orientation = RewardTermCfg(
        func=f"{_MDP}.rewards:root_orientation_tracking", weight=0.5,
        params={"std": 0.4},
    )
    action_rate = RewardTermCfg(func=f"{_MDP}.rewards:action_rate_penalty", weight=-0.1)
    joint_limit = RewardTermCfg(func=f"{_MDP}.rewards:joint_limit_penalty", weight=-2.0)


@configclass
class SuperMimicTerminationCfg:
    height = TerminationTermCfg(
        func=f"{_MDP}.terminations:height_termination", params={"threshold": 0.20},
    )
    tilt = TerminationTermCfg(
        func=f"{_MDP}.terminations:tilt_termination", params={"threshold": 0.8},
    )
    motion_end = TerminationTermCfg(func=f"{_MDP}.terminations:motion_end", time_out=True)
    time_out = TerminationTermCfg(func=f"{_MDP}.terminations:time_out", time_out=True)
    ankle: TerminationTermCfg | None = None
    wrist: TerminationTermCfg | None = None


@configclass
class SuperMimicEventCfg:
    # The reset event precedes the command's state write. All pose perturbations
    # belong to that write, so a later write cannot erase an event's noise.
    randomize_physics = EventTermCfg(
        func=f"{_MDP}.events:randomize_physics_parameters", mode="reset",
    )


@configclass
class SuperMimicCurriculumCfg:
    # Statistics are updated once per vector step, including ranks with no
    # resets. The reset curriculum reports them; it must not update them again.
    update_bin_returns = CurriculumTermCfg(func=f"{_MDP}.curriculum:update_motion_bin_returns")


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
                static_friction=0.3, dynamic_friction=0.3, restitution=0.0,
                friction_combine_mode="max", restitution_combine_mode="max",
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
            from .g1_robot import make_g1_robot_cfg
            self.scene.robot, self.action_scale = make_g1_robot_cfg(self.robot_usd, self.robot_urdf)
            self.actions.joint_pos.action_dim = self.action_dim
            if self.ground_usd:
                self.scene.ground.spawn.usd_path = self.ground_usd
            return
        self.scene.robot, self.action_scale = make_robot_cfg(
            actuator_type=self.actuator_type, min_delay=self.min_delay,
            max_delay=self.max_delay, neck_yaw=self.neck_yaw, neck_pitch=self.neck_pitch,
            **({"usd_path": self.robot_usd} if self.robot_usd is not None else {}),
        )
        for actuator_cfg in self.scene.robot.actuators.values():
            if isinstance(actuator_cfg, DelayedTorqueSpeedPDActuatorCfg):
                actuator_cfg.class_type = SuperMimicDelayedPDActuator


class SuperMimicRewardManager(RewardManager):
    """Capture the physical transition before the base environment's autoreset."""

    def compute(self, dt: float) -> torch.Tensor:
        reward = super().compute(dt)
        self._env._finish_transition(reward)
        return reward


class SuperMimicTerminationManager(TerminationManager):
    """Keep native raw terms; only accepted tracking draws terminate an episode."""

    def __init__(self, cfg, env):
        from .recovery import TRACKING_GROUPS

        super().__init__(cfg, env)
        self._tracking_indices = [self._term_name_to_term_idx[name] for name in TRACKING_GROUPS]
        self._tracking_ids = torch.tensor(self._tracking_indices, device=self.device)
        self.accepted_tracking = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)

    def compute(self):
        from .recovery import sample_tracking_terminations

        self._truncated_buf.zero_()
        self._terminated_buf.zero_()
        for index, cfg in enumerate(self._term_cfgs):
            value = cfg.func(self._env, **cfg.params)
            self._term_dones[:, index] = value
            if cfg.time_out:
                self._truncated_buf |= value
            elif index not in self._tracking_indices:
                self._terminated_buf |= value
        # Lab increments the episode counter before this call. Errors already
        # use the physical post-state and command.next_reference_batch().
        raw = self._term_dones[:, self._tracking_ids]
        if self._env.cfg.eval_mode:
            self.accepted_tracking.copy_(raw)
        else:
            self.accepted_tracking.copy_(sample_tracking_terminations(
                raw, self._env.recovery_role, self._env.episode_length_buf - 1,
                dt=self._env.step_dt,
            ))
        self._terminated_buf |= self.accepted_tracking.any(-1)
        accepted = self._term_dones.clone()
        accepted[:, self._tracking_ids] = self.accepted_tracking
        self._last_episode_dones.copy_(torch.where(self.dones[:, None], accepted, self._last_episode_dones))
        return self.dones


class SuperMimicRLEnv(ManagerBasedRLEnv):
    cfg: SuperMimicEnvCfg

    def __init__(self, cfg, *, motion=None, mdp=None, reference=None, distributed=None, **kwargs):
        command = cfg.commands.motion
        command.reset_noise = cfg.reset_noise
        command.eval_mode = cfg.eval_mode
        command.recovery_enabled = cfg.recovery_enabled
        if cfg.recovery_enabled:
            cfg.terminations.ankle = TerminationTermCfg(func=f"{_MDP}.terminations:ankle_termination")
            cfg.terminations.wrist = TerminationTermCfg(func=f"{_MDP}.terminations:wrist_termination")
        if motion is None and mdp is not None:
            motion = mdp.motion
        if motion is None and reference is not None:
            motion = reference.catalog
        if motion is None:
            urdf = command.urdf_path or str(
                Path(ASSET_DIR) / "rx_p2_29dof_20260824/rx_p2_29dof_20260824_full.urdf"
            )
            if cfg.robot_name == "g1":
                from .g1 import G1Kinematics
                kin = G1Kinematics(cfg.robot_urdf, cfg.sim.device)
            else:
                kin = RXP2Kinematics(urdf, cfg.sim.device)
            motion = (
                MotionLib.from_yaml(command.motion_file, kin, cfg.sim.device)
                if command.motion_file else MotionLib(command.motion_paths, kin, cfg.sim.device)
            )
        if reference is None:
            reference = mdp.reference if mdp is not None else StaticReference(motion)
        if reference.catalog is not motion:
            raise ValueError("Environment and reference must share the same motion catalog")
        if mdp is not None and (mdp.motion is not motion or mdp.reference is not reference):
            raise ValueError("Environment and MDP must share motion and reference")
        self.motion = motion
        self.reference = reference
        # Motion files have 27 policy joints. Match the two padded neck entries
        # to the independent servos; none of the tracked bodies is a head link.
        if cfg.robot_name == "rx_p2":
            motion.q[:, 27:] = motion.q.new_tensor((cfg.neck_yaw, cfg.neck_pitch))
            motion.qd[:, 27:] = 0.0
        if motion.action_dim != cfg.action_dim:
            raise ValueError("Motion and environment action dimensions must match")
        self.mdp = mdp or MDP(
            motion.kin, motion, torch.zeros(NUM_J, device=cfg.sim.device),
            RewardCfg(
                term_height=cfg.terminations.height.params["threshold"],
                term_tilt=cfg.terminations.tilt.params["threshold"],
            ),
            recovery_enabled=cfg.recovery_enabled,
            reference=reference,
        )
        if self.mdp.recovery_enabled != cfg.recovery_enabled:
            raise ValueError("Environment and MDP recovery_enabled must match")
        self.distributed = distributed
        self._post_state = None
        self._post_bodies = None
        self._tracking_errors = None
        self._reward_terms = None
        self.transition = None
        self._stepping = False
        # One analytic MDP serves real rewards and differentiable imagination.
        for name, width in (
            ("body_position", "body_pos_std"), ("body_orientation", "body_ori_std"),
            ("body_linvel", "body_linvel_std"), ("body_angvel", "body_angvel_std"),
            ("root_position", "root_pos_std"), ("root_orientation", "root_ori_std"),
        ):
            getattr(cfg.rewards, name).params["std"] = getattr(self.mdp.cfg, width)
        cfg.rewards.action_rate.weight = self.mdp.cfg.w_action_rate
        cfg.rewards.joint_limit.weight = self.mdp.cfg.w_joint_limit
        cfg.terminations.height.params["threshold"] = self.mdp.cfg.term_height
        cfg.terminations.tilt.params["threshold"] = self.mdp.cfg.term_tilt
        cfg.observations.supermimic_relative_orientation = (
            SuperMimicObservationsCfg.RelativeOrientationCfg(enable_corruption=not cfg.eval_mode)
            if cfg.actor_relative_orientation else None
        )
        if cfg.eval_mode:
            cfg.observations.supermimic_obs.enable_corruption = False
        self._configure_scene(cfg)
        super().__init__(cfg, **kwargs)

    def _configure_scene(self, cfg):
        pass

    def load_managers(self):
        self.xi = torch.zeros(self.num_envs, 33, device=self.device)
        self.action_scale = torch.tensor(self.cfg.action_scale, device=self.device)
        robot = self.scene["robot"]
        torso_id = robot.find_bodies(self.cfg.torso_body_name, preserve_order=True)[0][0]
        self.randomization = EpisodeRandomization(robot, self.xi, torso_id, eval_mode=self.cfg.eval_mode,
                                                   physics_manager=self.sim.physics_manager if getattr(self.cfg, "backend", None) == "newton" else None)
        self._previous_action = torch.zeros(self.num_envs, self.cfg.action_dim, device=self.device)
        super().load_managers()
        # Preserve Lab's manager initialization and physics loop. Only the
        # task-specific post-reward/pre-reset hook differs.
        self.reward_manager = SuperMimicRewardManager(self.cfg.rewards, self)
        if self.recovery_enabled:
            self.termination_manager = SuperMimicTerminationManager(self.cfg.terminations, self)

    def state(self):
        return canonical_state(self)

    def reference_address(self):
        return self.command_manager.get_term("motion").reference_address()

    def reference_batch(self, goal_frames=0, bodies=False):
        return self.command_manager.get_term("motion").reference_batch(goal_frames=goal_frames, bodies=bodies)

    def post_state(self):
        if self._post_state is None:
            self._post_state = self.state()
        return self._post_state

    def post_bodies(self):
        if self._post_bodies is None:
            self._post_bodies = self.mdp.kin.forward(*self.post_state())
        return self._post_bodies

    def tracking_errors(self):
        if self._tracking_errors is None:
            self._tracking_errors = self.mdp.tracking_errors(
                self.post_state(), self.command_manager.get_term("motion").next_reference_batch(bodies=True),
                body_state=self.post_bodies(),
            )
        return self._tracking_errors

    def reward_terms(self):
        if self._reward_terms is None:
            _, self._reward_terms = self.mdp.reward(
                self.post_state(), self.command_manager.get_term("motion").next_reference_batch(bodies=True),
                self.action_manager.action, self._previous_action, terms=True,
                **({"body_state": self.post_bodies()} if self.recovery_enabled else {}),
            )
        return self._reward_terms

    def step(self, action):
        action = clip_action(action)
        self._post_state = None
        self._post_bodies = None
        self._tracking_errors = None
        self._reward_terms = None
        self._previous_action.copy_(self.action_manager.action)
        ref_id, ref_phase = (value.clone() for value in self.reference_address())
        scene_id = self.reference.sample(ref_id, ref_phase).scene_id.clone()
        self.transition = {
            "s": flatten_state(self.state()), "a_prev": self._previous_action.clone(),
            "f": self.f.clone(), "xi": self.xi.clone(), "ep_step": self.episode_length_buf.clone(),
            "ref_id": ref_id, "ref_phase": ref_phase, "scene_id": scene_id,
        }
        controller = self.command_manager.get_term("motion").plan_controller
        if controller is not None:
            self.transition["generated_source"] = controller.enabled.clone()
        if self.recovery_enabled:
            self.transition["recovery_role"] = self.recovery_role.clone()
        self._capture_start_perception()
        self._stepping = True
        try:
            return super().step(action)
        finally:
            self._stepping = False

    def _reset_idx(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        # Evaluation records failures but continues the same physical rollout.
        # Explicit reset/reset_frames still initialize the requested trials.
        if self.cfg.eval_mode and self._stepping:
            return
        super()._reset_idx(env_ids)
        self._reset_sensors(env_ids)

    def reset(self, seed=None, env_ids=None, options=None):
        # The base reset writes ALL articulations, including non-reset rows.
        # Suppress only delay advancement; keep Lab's reset/render/obs lifecycle.
        actuators = [actuator for actuator in self.scene["robot"].actuators.values()
                     if isinstance(actuator, SuperMimicDelayedPDActuator)]
        for actuator in actuators:
            actuator.reset_sync_only = True
        try:
            return super().reset(seed=seed, env_ids=env_ids, options=options)
        finally:
            for actuator in actuators:
                actuator.reset_sync_only = False

    def _reset_sensors(self, env_ids):
        pass

    def write_root_link_velocity(self, velocity, env_ids):
        robot = self.scene["robot"]
        robot.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        # The pinned PhysX backend requires its converted COM buffer to be
        # submitted after the link setter. Keep the baseline correction.
        robot.write_root_com_velocity_to_sim_index(
            root_velocity=robot.data.root_com_vel_w.torch[env_ids].clone(), env_ids=env_ids,
        )

    def _finish_transition(self, reward):
        terminated = self.termination_manager.terminated.clone()
        truncated = self.termination_manager.time_outs.clone()
        self.transition.update(
            s1=flatten_state(self.post_state()), a=self.action_manager.action.clone(),
            r=reward.clone(), term=terminated, done=terminated | truncated,
            terms=self.reward_terms(),
        )
        self._capture_end_perception()
        metrics = self.command_manager.get_term("motion").record_step(
            reward, terminated, truncated, self.reward_terms()["track"]
        )
        if metrics:
            self.transition.update(metrics)

    def _capture_start_perception(self):
        pass

    def _capture_end_perception(self):
        pass

    def bind_motion_terrain(self, motion_ids, env_ids):
        pass

    def reset_frames(self, env_ids, frames):
        self.command_manager.get_term("motion").set_pending_frames(frames)
        return self.reset(env_ids=env_ids)

    @property
    def f(self):
        return self.command_manager.get_term("motion").time_steps

    @property
    def recovery_role(self):
        return self.command_manager.get_term("motion").recovery_role

    @property
    def recovery_enabled(self):
        return self.cfg.recovery_enabled

    @property
    def a_prev(self):
        return self.action_manager.action

    @property
    def default_q(self):
        return self.command_manager.get_term("motion").default_q

    @property
    def sampler(self):
        return self.command_manager.get_term("motion").sampler


@configclass
class SuperMimicErpEnvCfg(SuperMimicEnvCfg):
    neck_pitch: float = NECK_PITCH_DOWN_29DEG
    erp_depth_noise_m: float = 0.05


class SuperMimicErpRLEnv(SuperMimicRLEnv):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            from .erp import close_raycast_backends

            try:
                close_raycast_backends(self)
            except Exception:
                pass
            raise

    def close(self):
        from .erp import close_raycast_backends

        try:
            close_raycast_backends(self)
        finally:
            # Kit can tick during shutdown while another UI owner still holds
            # this controller. Unsubscribe before the base close deletes scene.
            controller = getattr(self, "viewport_camera_controller", None)
            handle = getattr(controller, "_viewport_camera_update_handle", None)
            if handle is not None:
                handle.unsubscribe()
                controller._viewport_camera_update_handle = None
            super().close()

    @property
    def camera_geometry(self):
        """Native RF mesh provider, forwarded by SuperMimicTrainingEnv."""
        from .geometry import ErpCameraGeometry
        geometry = getattr(self, "_supermimic_camera_geometry", None)
        if geometry is None:
            geometry = self._supermimic_camera_geometry = ErpCameraGeometry(
                self.scene.sensors["erp_camera"], self.scene.terrain, self.motion,
            )
        return geometry

    @property
    def camera_transport(self):
        """The real capture transport, also exposed through the training adapter."""
        from .erp import _camera_transport
        return _camera_transport(self)

    def _configure_scene(self, cfg):
        from .erp import configure_erp_scene
        configure_erp_scene(cfg, self.motion)

    def bind_motion_terrain(self, motion_ids, env_ids):
        from .erp import bind_terrain
        bind_terrain(self, motion_ids, env_ids)

    def perception(self):
        from .erp import capture_perception
        return capture_perception(self)

    def _capture_start_perception(self):
        if self.cfg.clean_perception:
            self.transition.update({key: value.clone() for key, value in self.perception().items()
                                    if key in ("depth", "scan", "hit", "clean_depth", "camera_frame", "camera_lag")})
            return
        self.transition.update({key: value.clone() for key, value in self.perception().items()
                                if key in ("depth", "scan", "hit")})

    def _capture_end_perception(self):
        if self.cfg.clean_perception:
            self.transition.update({key + "1": value.clone() for key, value in self.perception().items()
                                    if key in ("depth", "scan", "hit", "clean_depth")})
            return
        self.transition.update({key + "1": value.clone() for key, value in self.perception().items()
                                if key in ("depth", "scan", "hit")})

    def _reset_sensors(self, env_ids):
        from .erp import reset_perception
        reset_perception(self, env_ids)


class SuperMimicTrainingEnv:
    """Expose model-based transitions without changing ManagerBasedRLEnv.step."""

    def __init__(self, env: SuperMimicRLEnv):
        self.env = env
        self.N = env.num_envs
        self.device = env.device

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def reference(self):
        return self.env.reference

    def reference_address(self):
        return self.env.reference_address()

    def reference_batch(self, goal_frames=0, bodies=False):
        return self.env.reference_batch(goal_frames=goal_frames, bodies=bodies)

    @property
    def ep_step(self):
        return self.env.episode_length_buf

    def step(self, action):
        self.env.step(action)
        return self.env.transition

    def policy_observation(self):
        return self.env.obs_buf["supermimic_obs"]

    def policy_relative_orientation(self):
        return self.env.obs_buf["supermimic_relative_orientation"]

    def reset(self, env_ids=None, frames=None):
        if frames is not None:
            return self.env.reset_frames(env_ids, frames)
        return self.env.reset(env_ids=env_ids)

    def close(self):
        self.env.close()
