"""Original ManagerBased manager_based responsibilities."""

from __future__ import annotations

import math

from pathlib import Path

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import TerminationTermCfg

from mimic_baseline.assets import ASSET_DIR

from mimic_baseline.tasks.supermimic.core.action_bounds import clip_action

from mimic_baseline.tasks.supermimic.envs.actuators import (
    SuperMimicDelayedPDActuator,
    G1DelayedPDActuator,
)

from mimic_baseline.tasks.supermimic.core.kinematics import RXP2Kinematics

from mimic_baseline.tasks.supermimic.envs.mdp.commands import canonical_state

from mimic_baseline.tasks.supermimic.core.mdp import (
    MDP,
    NUM_J,
    RewardCfg,
    flatten_state,
)

from mimic_baseline.tasks.supermimic.data.motion import MotionLib

from mimic_baseline.tasks.supermimic.envs.randomization import EpisodeRandomization

from mimic_baseline.tasks.supermimic.data.reference import StaticReference

from mimic_baseline.tasks.supermimic.envs.configs.base import (
    SuperMimicObservationsCfg,
    SuperMimicEnvCfg,
)
from mimic_baseline.tasks.supermimic.envs.configs.base import _MDP
from mimic_baseline.tasks.supermimic.envs.managers import (
    SuperMimicRewardManager,
    SuperMimicTerminationManager,
)


class SuperMimicRLEnv(ManagerBasedRLEnv):
    cfg: SuperMimicEnvCfg

    def __init__(
        self, cfg, *, motion=None, mdp=None, reference=None, distributed=None, **kwargs
    ):
        command = cfg.commands.motion
        command.reset_noise = cfg.reset_noise
        command.eval_mode = cfg.eval_mode
        command.recovery_enabled = cfg.recovery_enabled
        if cfg.recovery_enabled:
            cfg.terminations.ankle = TerminationTermCfg(
                func=f"{_MDP}.terminations:ankle_termination"
            )
            cfg.terminations.wrist = TerminationTermCfg(
                func=f"{_MDP}.terminations:wrist_termination"
            )
        if motion is None and mdp is not None:
            motion = mdp.motion
        if motion is None and reference is not None:
            motion = reference.catalog
        if motion is None:
            urdf = command.urdf_path or str(
                Path(ASSET_DIR) / "rx_p2_29dof_20260824/rx_p2_29dof_20260824_full.urdf"
            )
            if cfg.robot_name == "g1":
                from mimic_baseline.tasks.supermimic.robots.g1.kinematics import (
                    G1Kinematics,
                )

                kin = G1Kinematics(cfg.robot_urdf, cfg.sim.device)
            else:
                kin = RXP2Kinematics(urdf, cfg.sim.device)
            motion = (
                MotionLib.from_yaml(command.motion_file, kin, cfg.sim.device)
                if command.motion_file
                else MotionLib(command.motion_paths, kin, cfg.sim.device)
            )
        if reference is None:
            reference = mdp.reference if mdp is not None else StaticReference(motion)
        if reference.catalog is not motion:
            raise ValueError(
                "Environment and reference must share the same motion catalog"
            )
        if mdp is not None and (
            mdp.motion is not motion or mdp.reference is not reference
        ):
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
            motion.kin,
            motion,
            torch.zeros(NUM_J, device=cfg.sim.device),
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
            ("body_position", "body_pos_std"),
            ("body_orientation", "body_ori_std"),
            ("body_linvel", "body_linvel_std"),
            ("body_angvel", "body_angvel_std"),
            ("root_position", "root_pos_std"),
            ("root_orientation", "root_ori_std"),
        ):
            getattr(cfg.rewards, name).params["std"] = getattr(self.mdp.cfg, width)
        cfg.rewards.action_rate.weight = self.mdp.cfg.w_action_rate
        cfg.rewards.joint_limit.weight = self.mdp.cfg.w_joint_limit
        cfg.terminations.height.params["threshold"] = self.mdp.cfg.term_height
        cfg.terminations.tilt.params["threshold"] = self.mdp.cfg.term_tilt
        cfg.observations.supermimic_relative_orientation = (
            SuperMimicObservationsCfg.RelativeOrientationCfg(
                enable_corruption=not cfg.eval_mode
            )
            if cfg.actor_relative_orientation
            else None
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
        torso_id = robot.find_bodies(self.cfg.torso_body_name, preserve_order=True)[0][
            0
        ]
        self.randomization = EpisodeRandomization(
            robot,
            self.xi,
            torso_id,
            eval_mode=self.cfg.eval_mode,
            physics_manager=self.sim.physics_manager
            if getattr(self.cfg, "backend", None) == "newton"
            else None,
        )
        self._previous_action = torch.zeros(
            self.num_envs, self.cfg.action_dim, device=self.device
        )
        super().load_managers()
        # Preserve Lab's manager initialization and physics loop. Only the
        # task-specific post-reward/pre-reset hook differs.
        self.reward_manager = SuperMimicRewardManager(self.cfg.rewards, self)
        if self.recovery_enabled:
            self.termination_manager = SuperMimicTerminationManager(
                self.cfg.terminations, self
            )

    def state(self):
        return canonical_state(self)

    def reference_address(self):
        return self.command_manager.get_term("motion").reference_address()

    def reference_batch(self, goal_frames=0, bodies=False):
        return self.command_manager.get_term("motion").reference_batch(
            goal_frames=goal_frames, bodies=bodies
        )

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
                self.post_state(),
                self.command_manager.get_term("motion").next_reference_batch(
                    bodies=True
                ),
                body_state=self.post_bodies(),
            )
        return self._tracking_errors

    def reward_terms(self):
        if self._reward_terms is None:
            _, self._reward_terms = self.mdp.reward(
                self.post_state(),
                self.command_manager.get_term("motion").next_reference_batch(
                    bodies=True
                ),
                self.action_manager.action,
                self._previous_action,
                terms=True,
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
            "s": flatten_state(self.state()),
            "a_prev": self._previous_action.clone(),
            "f": self.f.clone(),
            "xi": self.xi.clone(),
            "ep_step": self.episode_length_buf.clone(),
            "ref_id": ref_id,
            "ref_phase": ref_phase,
            "scene_id": scene_id,
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
        actuators = [
            actuator
            for actuator in self.scene["robot"].actuators.values()
            if isinstance(actuator, (SuperMimicDelayedPDActuator, G1DelayedPDActuator))
        ]
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
        robot.write_root_link_velocity_to_sim_index(
            root_velocity=velocity, env_ids=env_ids
        )
        # The pinned PhysX backend requires its converted COM buffer to be
        # submitted after the link setter. Keep the baseline correction.
        robot.write_root_com_velocity_to_sim_index(
            root_velocity=robot.data.root_com_vel_w.torch[env_ids].clone(),
            env_ids=env_ids,
        )

    def _finish_transition(self, reward):
        terminated = self.termination_manager.terminated.clone()
        truncated = self.termination_manager.time_outs.clone()
        self.transition.update(
            s1=flatten_state(self.post_state()),
            a=self.action_manager.action.clone(),
            r=reward.clone(),
            term=terminated,
            done=terminated | truncated,
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


# Keep the established module API while loading optional perception only on demand.
from .adapter import SuperMimicTrainingEnv


def __getattr__(name):
    if name in ("SuperMimicErpEnvCfg", "SuperMimicErpRLEnv"):
        from . import perception

        return getattr(perception, name)
    raise AttributeError(name)


from .configs.base import (
    PHYSICS_DT,
    DECIMATION,
    ENV_SPACING,
    MAX_EPISODE_LENGTH_S,
    NECK_PITCH_DOWN_29DEG,
    SuperMimicObservationsCfg,
    SuperMimicActionCfg,
    SuperMimicCommandCfg,
    SuperMimicRewardCfg,
    SuperMimicTerminationCfg,
    SuperMimicEventCfg,
    SuperMimicCurriculumCfg,
    SuperMimicSceneCfg,
    SuperMimicEnvCfg,
)
