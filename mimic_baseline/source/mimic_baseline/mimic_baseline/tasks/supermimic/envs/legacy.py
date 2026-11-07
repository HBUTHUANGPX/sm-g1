"""Batched PhysX motion-tracking environment for RX-P2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from mimic_baseline.assets import ASSET_DIR

from mimic_baseline.tasks.supermimic.core.action_bounds import clip_action
from mimic_baseline.tasks.supermimic.core.kinematics import RXP2_JOINT_NAMES
from mimic_baseline.tasks.supermimic.core.mdp import ACT_DIM, MAX_STEP_REWARD, MDP, flatten_state
from mimic_baseline.tasks.supermimic.data.motion import BIN_FRAMES, CONTROL_DT, MotionLib
from mimic_baseline.tasks.supermimic.envs.randomization import EpisodeRandomization
from mimic_baseline.tasks.supermimic.robots.rxp2.asset import make_robot_cfg
from mimic_baseline.tasks.supermimic.core.rotations import mat_to_quat, quat_to_mat, so3_exp


PHYSICS_DT = 0.005
DECIMATION = 4
ENV_SPACING = 2.5
GROUND_PLANE_USD = str(Path(ASSET_DIR) / "terrains/isaac_sim_6_grid/default_environment.usd")


def make_sim(device: str, args: Any = None):
    """Create the RX-P2 PhysX simulation config at an explicit 200 Hz."""
    del args
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}.")
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxCfg

    return sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        physics=PhysxCfg(),
    )


def _make_scene_cfg(num_envs: int, robot_cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    ground_material = sim_utils.PhysxRigidBodyMaterialCfg(
        static_friction=0.3,
        dynamic_friction=0.3,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="max",
    )

    @configclass
    class TrackingSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(usd_path=GROUND_PLANE_USD, physics_material=ground_material),
        )
        robot = robot_cfg

    return TrackingSceneCfg(
        num_envs=num_envs,
        env_spacing=ENV_SPACING,
        replicate_physics=True,
    )


class TrackingSim:
    """PhysX wrapper with 50 Hz policy steps and four 5 ms physics ticks."""

    def __init__(
        self,
        sim_cfg,
        num_envs: int,
        motion: MotionLib,
        mdp: MDP,
        gamma: float = 0.99,
        eval_mode: bool = False,
        max_episode_steps: int = 500,
        reset_noise: bool = True,
        distributed: Any | None = None,
        actuator_type: str = "delayed_torque_speed",
        max_delay: int = 4,
        *,
        reference=None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must lie in (0,1), got {gamma}.")

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from isaaclab.sim.utils.stage import use_stage

        self.device = sim_cfg.device
        self.N = num_envs
        self.motion = motion
        self.mdp = mdp
        self.reference = mdp.reference if reference is None else reference
        if mdp.motion is not motion or self.reference.catalog is not motion:
            raise ValueError("Environment and MDP must share the same motion catalog")
        if mdp.reference is not self.reference:
            raise ValueError("Environment and MDP must share the same reference")
        self.distributed = distributed
        self.gamma = gamma
        self.eval_mode = eval_mode
        self.reset_noise = bool(reset_noise and not eval_mode)
        self.max_episode_steps = (
            math.inf if eval_mode else int(max_episode_steps)
        )

        robot_cfg, alpha_values = make_robot_cfg(
            actuator_type, min_delay=0, max_delay=max_delay
        )

        self.sim = SimulationContext(sim_cfg)
        with use_stage(self.sim.stage):
            self.scene = InteractiveScene(_make_scene_cfg(num_envs, robot_cfg))
        self.sim.register_interactive_scene(self.scene)
        with use_stage(self.sim.stage):
            self.sim.reset()
        self.scene.update(0.0)
        self.robot = self.scene["robot"]

        joint_ids, joint_names = self.robot.find_joints(
            list(RXP2_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != RXP2_JOINT_NAMES:
            raise RuntimeError(
                "RX-P2 joint mapping differs from the canonical 29-joint order: "
                f"{tuple(joint_names)!r}."
            )
        self.joint_ids = torch.tensor(
            joint_ids, dtype=torch.int32, device=self.device
        )
        self.body_ids, _ = self.robot.find_bodies(
            list(mdp.kin.tracked_body_names), preserve_order=True
        )
        self.default_q = self.robot.data.default_joint_pos.torch[
            0, self.joint_ids
        ].clone()
        self.alpha = torch.tensor(
            alpha_values, dtype=torch.float32, device=self.device
        )
        self.torso_id = self.robot.find_bodies(
            "waist_pitch_link", preserve_order=True
        )[0][0]

        device = torch.device(self.device)
        self.xi = torch.zeros((num_envs, 33), dtype=torch.float32, device=device)
        self.randomization = EpisodeRandomization(
            self.robot, self.xi, self.torso_id, eval_mode=eval_mode
        )

        self.f = torch.zeros(
            (num_envs,), dtype=torch.long, device=device
        )
        self.ep_step = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(
            (num_envs, ACT_DIM), dtype=torch.float32, device=device
        )
        self.start_bin = torch.zeros_like(self.f)
        self.ep_disc_ret = torch.zeros((num_envs,), device=device)
        self.ep_ret = torch.zeros_like(self.ep_disc_ret)
        self.max_discounted_return = MAX_STEP_REWARD / (1.0 - gamma)
        self.bin_G = torch.zeros(
            (motion.num_bins,), dtype=torch.float32, device=device
        )
        self.bin_w_avg = torch.full(
            (motion.num_bins,),
            1.0 / motion.num_bins,
            dtype=torch.float32,
            device=device,
        )
        self.done_eval = torch.zeros(
            (num_envs,), dtype=torch.bool, device=device
        )
        self.failed_eval = torch.zeros_like(self.done_eval)
        self.reset(torch.arange(num_envs, device=device))

    def state(self):
        """Return the current canonical state with environment-local root position."""
        data = self.robot.data
        pose = data.root_link_pose_w.torch.clone()
        velocity = data.root_link_vel_w.torch.clone()
        position = pose[:, :3] - self.scene.env_origins
        quaternion_xyzw = pose[:, 3:7]
        rotation = quat_to_mat(
            torch.cat((quaternion_xyzw[:, 3:4], quaternion_xyzw[:, :3]), dim=-1)
        )
        joint_pos = data.joint_pos.torch[:, self.joint_ids].clone()
        joint_vel = data.joint_vel.torch[:, self.joint_ids].clone()
        return position, rotation, velocity[:, :3], velocity[:, 3:6], joint_pos, joint_vel

    def reference_address(self):
        """Translate static global-frame bookkeeping into a provider address."""
        return self.reference.locate(self.f)

    def reference_batch(self, goal_frames=0, bodies=False):
        plan_ids, offsets = self.reference_address()
        return self.reference.sample(plan_ids, offsets, goal_frames=goal_frames, bodies=bodies)

    def _advance_bin_history(self, active: torch.Tensor) -> None:
        soft = torch.softmax(
            -25.0 * self.bin_G / self.max_discounted_return, dim=0
        )
        candidate = self.bin_w_avg.mul(0.999).add(0.001 * soft)
        self.bin_w_avg.copy_(torch.where(active, candidate, self.bin_w_avg))

    def _reset_probabilities(self, *, advance_history: bool = True) -> torch.Tensor:
        soft = torch.softmax(
            -25.0 * self.bin_G / self.max_discounted_return, dim=0
        )
        if advance_history:
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        past = self.bin_w_avg / self.bin_w_avg.sum()
        return (
            0.5 / self.motion.num_bins
            + 0.4 * soft
            + 0.1 * past
        )

    def sample_frames(
        self,
        count: int,
        *,
        advance_history: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample return-biased frames with no start at or beyond clip end."""
        bins = torch.multinomial(
            self._reset_probabilities(advance_history=advance_history),
            count,
            replacement=True,
        )
        start = self.motion.bin_start[bins]
        end = torch.minimum(
            start + BIN_FRAMES,
            self.motion.clip_end[self.motion.bin_clip[bins]],
        )
        frames = start + (
            torch.rand(count, device=start.device) * (end - start).float()
        ).long()
        return frames, bins

    def _update_bin_returns(
        self,
        bins: torch.Tensor,
        returns: torch.Tensor,
        distributed: Any | None = None,
        reset_count: int = 0,
    ) -> torch.Tensor | None:
        """EMA-update bins from globally pooled sample sums and counts."""
        distributed = distributed if distributed is not None else self.distributed
        enabled = bool(getattr(distributed, "enabled", False))
        if bins.numel() == 0 and not enabled:
            return None
        bin_sums = torch.zeros(
            (self.motion.num_bins,), dtype=returns.dtype, device=returns.device
        )
        bin_counts = torch.zeros_like(bin_sums)
        if bins.numel() > 0:
            bin_sums.index_add_(0, bins, returns)
            bin_counts.index_add_(0, bins, torch.ones_like(returns))
        if enabled:
            packed = torch.cat(
                (
                    bin_sums,
                    bin_counts,
                    torch.tensor(
                        reset_count,
                        dtype=returns.dtype,
                        device=returns.device,
                    ).reshape(1),
                )
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            bin_sums = packed[: self.motion.num_bins]
            bin_counts = packed[
                self.motion.num_bins : 2 * self.motion.num_bins
            ]
            global_resets = packed[-1]
        else:
            global_resets = None
        visited = bin_counts > 0
        means = bin_sums / bin_counts.clamp_min(1.0)
        updated = 0.8 * self.bin_G + 0.2 * means
        self.bin_G.copy_(torch.where(visited, updated, self.bin_G))
        return global_resets

    def reset(
        self,
        env_ids: torch.Tensor,
        frames: torch.Tensor | None = None,
        *,
        advance_history: bool = True,
    ) -> None:
        """Reset selected environments without integrating an extra physics step."""
        if env_ids.numel() == 0:
            return
        # IsaacLab's indexed Warp writes consume int32, without dtype conversion.
        env_ids = env_ids.to(dtype=torch.int32)
        count = env_ids.numel()
        if frames is None:
            frames, bins = self.sample_frames(
                count, advance_history=advance_history
            )
        else:
            frames = frames.to(device=self.device, dtype=torch.long)
            bins = self.motion.frame_bin[frames]

        plan_ids, offsets = self.reference.locate(frames)
        reference = self.reference.sample(plan_ids, offsets)
        p, rotation, velocity, angular_velocity, joint_pos, joint_vel = (
            value.clone() for value in reference.now.state
        )
        if self.reset_noise:
            p = p + torch.stack(
                (
                    torch.empty(count, device=p.device).uniform_(-0.05, 0.05),
                    torch.empty(count, device=p.device).uniform_(-0.05, 0.05),
                    torch.empty(count, device=p.device).uniform_(-0.01, 0.01),
                ),
                dim=-1,
            )
            rotation_offset = torch.stack(
                (
                    torch.empty(count, device=p.device).uniform_(-0.1, 0.1),
                    torch.empty(count, device=p.device).uniform_(-0.1, 0.1),
                    torch.empty(count, device=p.device).uniform_(-0.2, 0.2),
                ),
                dim=-1,
            )
            rotation = so3_exp(rotation_offset) @ rotation
            velocity = velocity + torch.stack(
                (
                    torch.empty(count, device=p.device).uniform_(-0.5, 0.5),
                    torch.empty(count, device=p.device).uniform_(-0.5, 0.5),
                    torch.empty(count, device=p.device).uniform_(-0.2, 0.2),
                ),
                dim=-1,
            )
            angular_velocity = angular_velocity + torch.stack(
                (
                    torch.empty(count, device=p.device).uniform_(-0.52, 0.52),
                    torch.empty(count, device=p.device).uniform_(-0.52, 0.52),
                    torch.empty(count, device=p.device).uniform_(-0.78, 0.78),
                ),
                dim=-1,
            )
            joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(
                -0.1, 0.1
            )
        joint_pos = torch.maximum(
            torch.minimum(joint_pos, self.mdp.kin.q_hi),
            self.mdp.kin.q_lo,
        )

        quaternion_wxyz = mat_to_quat(rotation)
        root_pose = torch.cat(
            (
                p + self.scene.env_origins[env_ids],
                quaternion_wxyz[:, 1:4],
                quaternion_wxyz[:, 0:1],
            ),
            dim=-1,
        )
        self.scene.reset(env_ids)
        self.randomization.resample(env_ids)
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=root_pose, env_ids=env_ids
        )
        self.write_root_link_velocity(torch.cat((velocity, angular_velocity), dim=-1), env_ids)
        self.robot.write_joint_position_to_sim_index(
            position=joint_pos,
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel,
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        target = self.default_q + self.xi[env_ids, 1:30]
        self.robot.set_joint_position_target_index(
            target=target,
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        self.refresh()

    def refresh(self) -> None:
        """Refresh tensor views after reset, without integrating physics."""
        self.scene.update(0.0)

    def perception(self) -> dict[str, torch.Tensor]:
        """Return the current ERP depth frame and its valid mask.

        Contract expected by :class:`~.training.Trainer` when
        ``config.perception`` is enabled::

            {"depth": [N, 40, 40] metres, "valid": [N, 40, 40] bool}

        ``depth`` carries the delayed, noisy LF measurement with invalid pixels
        at the far clip, exactly as ``erp_observations.erp_depth`` supplies it
        in the tracking task. ``valid`` is the fixed stereo mask; it is not the
        per-frame occlusion mask, which the encoder does not consume.

        This environment has no ERP sensor yet, so the method refuses rather
        than returning a constant or zeros: a fabricated frame would train the
        encoder and the world model on data that does not exist, and the loss
        curve would look healthy while the branch learned nothing.
        """
        raise NotImplementedError(
            "TrackingSim has no ERP depth sensor. Enabling config.perception "
            "requires the manager-based environment with the tracking ERP "
            "sensor mounted on neck_pitch_link and the motion terrain; see "
            "docs/research/supermimic-perception-terrain-handoff-20261022.md."
        )

    def write_root_link_velocity(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Keep link/COM caches and PhysX consistent on the pinned Lab revision."""
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        # Lab's link setter computes both buffers, but passes the link buffer to
        # PhysX's COM setter. Resubmit its correctly converted COM buffer.
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=self.robot.data.root_com_vel_w.torch[env_ids].clone(),
            env_ids=env_ids,
        )

    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Execute one 50 Hz policy action as four explicit-PD physics ticks."""
        if action.shape != (self.N, ACT_DIM):
            raise ValueError(
                f"Expected action shape {(self.N, ACT_DIM)}, got {tuple(action.shape)}."
            )
        action = clip_action(action)
        state = self.state()
        frame = self.f.clone()
        ref_id, ref_phase = (value.clone() for value in self.reference_address())
        scene_id = self.reference.sample(ref_id, ref_phase).scene_id.clone()
        xi = self.xi.clone()
        previous_action = self.a_prev.clone()
        episode_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30]
        target[:, :27] += self.alpha * action
        self.robot.set_joint_position_target_index(
            target=target,
            joint_ids=self.joint_ids,
            env_ids=None,
        )
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)

        next_state = self.state()
        reference = self.reference.sample(ref_id, ref_phase + 1, bodies=True)
        next_frame = reference.source_frame
        reward, reward_terms = self.mdp.reward(
            next_state,
            reference,
            action,
            previous_action,
            terms=True,
        )
        terminated = self.mdp.termination(next_state, reference)
        next_episode_step = episode_step + 1
        timeout = reference.expiry.episode_end | (
            next_episode_step >= self.max_episode_steps
        )
        done = terminated | timeout
        output = {
            "s": flatten_state(state),
            "s1": flatten_state(next_state),
            "a": action.clone(),
            "a_prev": previous_action,
            "f": frame,
            "ref_id": ref_id,
            "ref_phase": ref_phase,
            "scene_id": scene_id,
            "xi": xi,
            "ep_step": episode_step,
            "term": terminated,
            "done": done,
            "r": reward,
            "terms": reward_terms,
        }

        self.ep_disc_ret.add_(
            self.gamma ** episode_step.float() * reward
        )
        self.ep_ret.add_(reward)
        self.f = next_frame
        self.ep_step = next_episode_step
        self.a_prev = action.clone()

        if self.eval_mode:
            self.failed_eval |= terminated & ~self.done_eval
            self.done_eval |= done
            return output

        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            lengths = self.ep_step[reset_ids].float()
            bootstrap = torch.where(
                terminated[reset_ids],
                torch.zeros_like(lengths),
                self.gamma**lengths
                * (self.ep_ret[reset_ids] / lengths)
                / (1.0 - self.gamma),
            )
            returns = self.ep_disc_ret[reset_ids] + bootstrap
            output["ep_len"] = lengths
            output["ep_ret_norm"] = (
                self.ep_ret[reset_ids] / (lengths * MAX_STEP_REWARD)
            )
        else:
            returns = torch.empty(0, dtype=self.ep_disc_ret.dtype, device=self.device)
        # Every rank enters this collective, including ranks with no resets.
        global_resets = self._update_bin_returns(
            self.start_bin[reset_ids],
            returns,
            reset_count=int(reset_ids.numel()),
        )
        distributed = self.distributed
        if bool(getattr(distributed, "enabled", False)):
            assert global_resets is not None
            self._advance_bin_history(global_resets > 0)
        if reset_ids.numel() > 0:
            self.reset(
                reset_ids,
                advance_history=not bool(
                    getattr(distributed, "enabled", False)
                ),
            )
        return output
