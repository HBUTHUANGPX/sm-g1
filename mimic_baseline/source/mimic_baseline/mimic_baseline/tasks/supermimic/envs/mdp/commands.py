"""SuperMimic reference playback and mastery-weighted bin sampling.

Static playback keeps :class:`~..motion.MotionLib` global frame indices for
sampler and replay bookkeeping. Reference values come from structured samples
of the provider shared by the environment and analytic MDP.

The command records bin attempts before autoreset. The sampler owns the bounded
distribution shared by real resets and the two replay sampling policies.
"""

from __future__ import annotations

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from mimic_baseline.tasks.supermimic.core.kinematics import RXP2_JOINT_NAMES
from mimic_baseline.tasks.supermimic.core.mdp import MAX_STEP_REWARD
from mimic_baseline.tasks.supermimic.data.sampling import MasterySampler


# Reset perturbations, mirroring the previous ``TrackingSim`` draws so a
# checkpoint trained on either path sees the same start distribution.
RESET_POS_XY = 0.05
RESET_POS_Z = 0.01
RESET_ROT = (0.1, 0.1, 0.2)
RESET_VEL_XY = 0.5
RESET_VEL_Z = 0.2
RESET_ANG_VEL = (0.52, 0.52, 0.78)
RESET_JOINT = 0.1


class SuperMimicMotionCommand(CommandTerm):
    """Reference playback, mastery-weighted reset sampling and terrain binding."""

    cfg: SuperMimicMotionCommandCfg

    def __init__(self, cfg: SuperMimicMotionCommandCfg, env):
        if not 0.0 < cfg.gamma < 1.0:
            raise ValueError(f"gamma must lie in (0,1), got {cfg.gamma}.")
        if cfg.goal_frames < 1:
            raise ValueError("goal_frames must be positive.")
        for name in ("motion", "reference", "mdp", "xi", "randomization"):
            if getattr(env, name, None) is None:
                raise ValueError(f"env.{name} must be initialized before the motion command.")
        super().__init__(cfg, env)
        self.motion = env.motion
        self.reference = env.reference
        self.static_reference = env.reference
        self.plan_controller = None
        self.mdp = env.mdp
        if self.motion.num_frames < 2:
            raise ValueError("SuperMimic requires at least two reference frames.")
        if self.mdp.motion is not self.motion or self.reference.catalog is not self.motion:
            raise ValueError("env.mdp and the command must share env.motion.")
        if self.mdp.reference is not self.reference:
            raise ValueError("env.mdp and the command must share env.reference.")

        self.robot = env.scene[cfg.asset_name]
        joint_ids, joint_names = self.robot.find_joints(
            list(self.mdp.kin.joint_names), preserve_order=True
        )
        if tuple(joint_names) != tuple(self.mdp.kin.joint_names):
            raise RuntimeError(
                "Robot joint mapping differs from the canonical 29-joint order: "
                f"{tuple(joint_names)!r}."
            )
        self.joint_ids = torch.tensor(joint_ids, dtype=torch.int32, device=self.device)
        self.default_q = self.robot.data.default_joint_pos.torch[0, self.joint_ids].clone()
        self.mdp.default_q = self.default_q
        # Reset events own randomization; the command only reads that same context.
        self.xi = env.xi
        self.randomization = env.randomization
        self.gamma = cfg.gamma
        self.eval_mode = cfg.eval_mode
        self.reset_noise = bool(cfg.reset_noise and not cfg.eval_mode)
        self.recovery_enabled = bool(cfg.recovery_enabled)
        self.recovery_role = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.recovery_enabled and self.reset_noise:
            from mimic_baseline.tasks.supermimic.core.recovery import RecoveryPlacement
            self._recovery_placement = RecoveryPlacement(
                self.mdp.kin, self.robot.cfg.spawn.usd_path,
                physics_variant="mujoco" if getattr(env.cfg, "backend", None) == "newton" else None,
            )
            self._recovery_roll_axis = self.default_q.new_tensor((1., 0., 0.))
            self._recovery_pitch_axis = self.default_q.new_tensor((0., 1., 0.))

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ep_ret = torch.zeros(self.num_envs, device=self.device)
        self._pending_frames: torch.Tensor | None = None
        self.last_episode_metrics: dict[str, torch.Tensor] = {}

        distributed = getattr(env, "distributed", None)
        self.sampler = MasterySampler(
            self.motion, self.num_envs, self.device,
            distributed=bool(getattr(distributed, "enabled", False)),
        )
        self.metrics["sample_bin"] = torch.zeros(self.num_envs, device=self.device)

    # -- sampling -------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Reference state for the current frame, shaped ``(N, 58)``."""
        now = self.reference_batch().now
        return torch.cat((now.q, now.qd), dim=-1)

    def reference_address(self):
        """Translate static global-frame bookkeeping into a provider address."""
        if self.plan_controller is not None:
            return self._reference_ids, self._reference_phases
        return self.reference.locate(self.time_steps)

    def install_plan_controller(self, controller):
        """Keep a pre-action clock independent of Lab's episode counter update."""
        self._reference_ids, self._reference_phases = self.static_reference.locate(self.time_steps)
        self.plan_controller = controller
        self.reference = controller.reference
        self._env.reference = self.reference
        self.mdp.reference = self.reference

    def reference_batch(self, goal_frames=0, bodies=False):
        """Sample the current static playback phase without advancing it."""
        plan_ids, offsets = self.reference_address()
        return self.reference.sample(plan_ids, offsets, goal_frames=goal_frames, bodies=bodies)

    def next_reference_batch(self, bodies=False):
        """Sample the post-physics phase before the command clock advances."""
        plan_ids, offsets = self.reference_address()
        return self.reference.sample(plan_ids, offsets + 1, bodies=bodies)

    def sample_frames(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sampler.sample_frames(count)

    def set_pending_frames(self, frames: torch.Tensor) -> None:
        """Seed the next reset with explicit frames (evaluation trials)."""
        frames = frames.to(device=self.device, dtype=torch.long)
        if frames.ndim != 1 or torch.any((frames < 0) | (frames >= self.motion.num_frames)):
            raise ValueError("Pending frames must be a vector of valid global frame indices.")
        self._pending_frames = frames.clone()

    # -- CommandTerm interface ------------------------------------------------

    def reset(self, env_ids=None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            selection = slice(None) if env_ids is None else env_ids
            env_ids = torch.arange(self.num_envs, device=self.device)[selection]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return {}
        if self._pending_frames is not None and self._pending_frames.numel() != env_ids.numel():
            raise ValueError("Pending frame count must match the reset environment count.")
        return super().reset(env_ids)

    def _resample_command(self, env_ids) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = env_ids.numel()
        if count == 0:
            return
        if self._pending_frames is not None:
            if self._pending_frames.numel() != count:
                raise ValueError("Pending frame count must match the reset environment count.")
            frames = self._pending_frames
            bins = self.motion.frame_bin[frames]
            self._pending_frames = None
        else:
            frames, bins = self.sample_frames(count)

        plan_ids, offsets = self.static_reference.locate(frames)
        motion_ids = self.static_reference.sample(plan_ids, offsets).scene_id
        self._env.bind_motion_terrain(motion_ids, env_ids)
        self._write_reset_state(frames, env_ids)

        self.time_steps[env_ids] = frames
        self.start_bin[env_ids] = bins
        self.ep_ret[env_ids] = 0.0
        self.sampler.reset(env_ids, frames)
        self.metrics["sample_bin"][env_ids] = bins.float()
        if self.plan_controller is not None:
            self.plan_controller.reset(env_ids)
            self._reference_ids[env_ids] = plan_ids
            self._reference_phases[env_ids] = offsets

    def _write_reset_state(
        self, frames: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        from mimic_baseline.tasks.supermimic.core.rotations import mat_to_quat, so3_exp

        plan_ids, offsets = self.static_reference.locate(frames)
        reference = self.static_reference.sample(plan_ids, offsets).now
        p, rotation, velocity, angular_velocity, joint_pos, joint_vel = (
            value.clone() for value in reference.state
        )
        count = env_ids.numel()
        device = p.device

        if self.reset_noise:
            p = p + torch.stack(
                (
                    torch.empty(count, device=device).uniform_(-RESET_POS_XY, RESET_POS_XY),
                    torch.empty(count, device=device).uniform_(-RESET_POS_XY, RESET_POS_XY),
                    torch.empty(count, device=device).uniform_(-RESET_POS_Z, RESET_POS_Z),
                ),
                dim=-1,
            )
            rotation_offset = torch.stack(
                (
                    torch.empty(count, device=device).uniform_(-RESET_ROT[0], RESET_ROT[0]),
                    torch.empty(count, device=device).uniform_(-RESET_ROT[1], RESET_ROT[1]),
                    torch.empty(count, device=device).uniform_(-RESET_ROT[2], RESET_ROT[2]),
                ),
                dim=-1,
            )
            rotation = so3_exp(rotation_offset) @ rotation
            velocity = velocity + torch.stack(
                (
                    torch.empty(count, device=device).uniform_(-RESET_VEL_XY, RESET_VEL_XY),
                    torch.empty(count, device=device).uniform_(-RESET_VEL_XY, RESET_VEL_XY),
                    torch.empty(count, device=device).uniform_(-RESET_VEL_Z, RESET_VEL_Z),
                ),
                dim=-1,
            )
            angular_velocity = angular_velocity + torch.stack(
                (
                    torch.empty(count, device=device).uniform_(-RESET_ANG_VEL[0], RESET_ANG_VEL[0]),
                    torch.empty(count, device=device).uniform_(-RESET_ANG_VEL[1], RESET_ANG_VEL[1]),
                    torch.empty(count, device=device).uniform_(-RESET_ANG_VEL[2], RESET_ANG_VEL[2]),
                ),
                dim=-1,
            )
            joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(-RESET_JOINT, RESET_JOINT)
        self.recovery_role[env_ids] = False
        if self.recovery_enabled and self.reset_noise:
            from mimic_baseline.tasks.supermimic.core.recovery import RECOVERY_PROBABILITY
            from mimic_baseline.tasks.supermimic.core.rotations import axis_angle_to_mat, yaw_mat

            role = torch.rand(count, device=device) < RECOVERY_PROBABILITY
            self.recovery_role[env_ids] = role
            roll_pitch = torch.empty(count, 2, device=device).uniform_(-torch.pi, torch.pi)
            recovered_rotation = (
                yaw_mat(reference.R)
                @ axis_angle_to_mat(self._recovery_pitch_axis, roll_pitch[:, 1])
                @ axis_angle_to_mat(self._recovery_roll_axis, roll_pitch[:, 0])
            )
            recovered_joints = reference.q + torch.empty_like(joint_pos).uniform_(-.2, .2)
            p = torch.where(role[:, None], reference.p, p)
            rotation = torch.where(role[:, None, None], recovered_rotation, rotation)
            joint_pos = torch.where(role[:, None], recovered_joints, joint_pos)
            velocity = torch.where(role[:, None], 0., velocity)
            angular_velocity = torch.where(role[:, None], 0., angular_velocity)
            joint_vel = torch.where(role[:, None], 0., joint_vel)
        joint_pos = torch.maximum(
            torch.minimum(joint_pos, self.mdp.kin.q_hi), self.mdp.kin.q_lo
        )
        # The neck is a fixed camera mount, independent of motion and calibration.
        action_dim = getattr(self.mdp.kin, "action_dim", 27)
        joint_pos[:, action_dim:] = self.default_q[action_dim:]
        joint_vel[:, action_dim:] = 0.0
        self.xi[env_ids, 1 + action_dim:30] = 0.0

        if self.recovery_enabled and self.reset_noise:
            origins = self._env.scene.env_origins[env_ids]
            terrain = getattr(self._env.scene, "terrain", None)
            geometry = {}
            if hasattr(terrain, "assignment"):
                scenes = terrain.assignment.scene_ids[env_ids]
                scene_origins = terrain.scene_origins[env_ids]
                geometry = {
                    "obstacles": terrain.obstacle_bounds[scenes] + scene_origins[:, None, None],
                    "floor_z": terrain.floor_z[scenes] + scene_origins[:, 2],
                }
            else:
                geometry["floor_z"] = origins[:, 2]
            candidate = self._recovery_placement.place(
                p + origins, rotation, joint_pos, **geometry,
            ) - origins
            p = torch.where(role[:, None], candidate, p)

        if not self.eval_mode:
            # Ordinary resets need clearance after pose noise; recovery already
            # places its collision bounds above the floor and terrain.
            p[:, 2] += (~self.recovery_role[env_ids]) * self.cfg.reset_height_offset

        quaternion_wxyz = mat_to_quat(rotation)
        root_pose = torch.cat(
            (
                p + self._env.scene.env_origins[env_ids],
                quaternion_wxyz[:, 1:4],
                quaternion_wxyz[:, 0:1],
            ),
            dim=-1,
        )
        # Indexed Warp writes consume int32 without converting torch int64 IDs.
        env_ids32 = env_ids.to(dtype=torch.int32)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids32)
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.cat((velocity, angular_velocity), dim=-1),
            env_ids=env_ids32,
        )
        # The pinned link setter computes COM velocity but submits the link buffer
        # to PhysX. Resubmit its converted COM buffer, as TrackingSim does.
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=self.robot.data.root_com_vel_w.torch[env_ids].clone(),
            env_ids=env_ids32,
        )
        self.robot.write_joint_position_to_sim_index(
            position=joint_pos,
            joint_ids=self.joint_ids,
            env_ids=env_ids32,
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel,
            joint_ids=self.joint_ids,
            env_ids=env_ids32,
        )
        self.robot.set_joint_position_target_index(
            target=self.default_q + self.xi[env_ids, 1:30],
            joint_ids=self.joint_ids,
            env_ids=env_ids32,
        )

    def _update_command(self) -> None:
        # Autoreset has installed new frames and zeroed episode lengths before
        # the base environment calls command_manager.compute().
        just_reset = self._env.episode_length_buf == 0
        if self.plan_controller is None:
            next_steps = self.next_reference_batch().source_frame
            self.time_steps.copy_(torch.where(just_reset, self.time_steps, next_steps))
        else:
            # This clock belongs only to the static catalog. A generated
            # provider needs no global frame metadata at all.
            plans, offsets = self.static_reference.locate(self.time_steps)
            next_steps = self.static_reference.sample(plans, offsets + 1).source_frame
            advance_static = ~just_reset & (self._reference_ids >= 0)
            self.time_steps.copy_(torch.where(advance_static, next_steps, self.time_steps))
        if self.plan_controller is not None:
            self._reference_phases.add_((~just_reset).long())

    def _update_metrics(self) -> None:
        self.metrics["sample_bin"][:] = self.start_bin.float()

    # -- algorithm-facing helpers --------------------------------------------

    def next_frame(self) -> torch.Tensor:
        """Static source-frame metadata for the post-physics reference."""
        return self.next_reference_batch().source_frame

    def at_clip_end(self) -> torch.Tensor:
        """Test the post-physics frame before CommandManager advances playback."""
        finished = self.next_reference_batch().expiry.episode_end
        if self.plan_controller is not None:
            finished = finished & ~self.plan_controller.enabled
        return finished

    def compute_goal(self, env) -> torch.Tensor:
        """Future motion reference window, shape (N, goal_frames * 69)."""
        return self.reference_batch(goal_frames=self.cfg.goal_frames).goal()

    def mean_bin_mastery(self) -> torch.Tensor:
        return self.sampler.mastery.mean()

    def mean_reset_frame(self) -> torch.Tensor:
        """Scalar mean of current frame indices (for logging)."""
        return self.time_steps.float().mean()

    @torch.no_grad()
    def record_step(
        self, reward: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor,
        tracking_reward: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Record one post-physics reward before autoreset, on every rank.

        The manager has already incremented ``episode_length_buf``. This method
        does not advance reference frames or reset episodes; ``compute`` and
        ``reset`` perform those operations after terminal transitions are saved.
        Curriculum terms must only read these statistics, never update them.
        """
        lengths = self._env.episode_length_buf.float()
        self.ep_ret.add_(reward)
        self.last_episode_metrics = {}
        if self.eval_mode:
            return self.last_episode_metrics

        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        episode_lengths = lengths[reset_ids]
        if reset_ids.numel() > 0:
            self.last_episode_metrics = {
                "ep_len": episode_lengths,
                "ep_ret_norm": self.ep_ret[reset_ids] / (episode_lengths * MAX_STEP_REWARD),
            }

        self.sampler.record_step(
            self.time_steps, (tracking_reward / MAX_STEP_REWARD).clamp(0.0, 1.0),
            terminated, truncated,
            **({"active_mask": ~self.plan_controller.enabled}
               if self.plan_controller is not None else {}),
        )
        return self.last_episode_metrics


@configclass
class SuperMimicMotionCommandCfg(CommandTermCfg):
    class_type: type = SuperMimicMotionCommand
    asset_name: str = "robot"
    # The environment wrapper resolves these when runtime objects are omitted.
    motion_paths: list[str] = []
    motion_file: str | None = None
    urdf_path: str | None = None
    step_dt: float = 0.02
    gamma: float = 0.99
    goal_frames: int = 10
    eval_mode: bool = False
    reset_noise: bool = True
    reset_height_offset: float = 0.20
    recovery_enabled: bool = False
    # resampling_time_range is required by CommandTermCfg; SuperMimicMotionCommand
    # uses mastery-weighted sampling rather than fixed resampling intervals, so we set
    # a large range and ignore it in _resample_command (the base class never fires
    # forced resampling inside an episode when the range is long).
    resampling_time_range: tuple[float, float] = (1e9, 1e9)


def canonical_state(env) -> tuple[torch.Tensor, ...]:
    """Return ``(p, R, v, w, q, qd)`` in the motion frame, matching ``TrackingSim``."""
    from mimic_baseline.tasks.supermimic.core.rotations import quat_to_mat

    command = env.command_manager.get_term("motion")
    robot = command.robot
    pose = robot.data.root_link_pose_w.torch.clone()
    velocity = robot.data.root_link_vel_w.torch.clone()
    position = pose[:, :3] - env.scene.env_origins
    quaternion_xyzw = pose[:, 3:7]
    rotation = quat_to_mat(torch.cat((quaternion_xyzw[:, 3:4], quaternion_xyzw[:, :3]), dim=-1))
    joint_pos = robot.data.joint_pos.torch[:, command.joint_ids].clone()
    joint_vel = robot.data.joint_vel.torch[:, command.joint_ids].clone()
    return position, rotation, velocity[:, :3], velocity[:, 3:6], joint_pos, joint_vel


__all__ = ["SuperMimicMotionCommand", "SuperMimicMotionCommandCfg", "canonical_state"]
