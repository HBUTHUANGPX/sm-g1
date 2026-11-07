from __future__ import annotations


import torch


from mimic_baseline.tasks.supermimic.core.action_bounds import (
    ACTION_LIMIT as _COLLECTION_ACTION_LIMIT,
    clip_action,
)


class CollectionMixin:
    """Original Trainer methods; state is owned by Trainer."""

    def _check_physics(self) -> None:
        if self.physics_health is not None:
            self.physics_health.check()

    def collect(self, steps: int) -> None:
        """Collect real simulator steps; all reductions stay on device until logging."""
        self._check_physics()
        if steps:
            self.cuda_cache.before_collection(self.iteration)
        with self.profiler.record("collection"):
            for _ in range(steps):
                state = self._as_state(self.env.state())
                xi = self.env.xi
                policy_observation = getattr(self.env, "policy_observation", None)
                if callable(policy_observation):
                    policy_obs = policy_observation()
                else:
                    policy_obs = self.mdp.add_noise(
                        self._obs(state, self.env.a_prev, xi)
                    )
                fresh = self.env.ep_step == 0
                if self._obs_history is None:
                    self._obs_history = (
                        policy_obs.unsqueeze(1).expand(-1, self.cfg.hist, -1).clone()
                    )
                else:
                    shifted = torch.cat(
                        (self._obs_history[:, 1:], policy_obs.unsqueeze(1)), 1
                    )
                    self._obs_history = torch.where(
                        fresh[:, None, None], policy_obs.unsqueeze(1), shifted
                    )
                if self.pmg is not None:
                    orientation_before_replan = (
                        self._relative_orientation(state, self.env.reference_batch())
                        if self.cfg.actor_relative_orientation
                        else None
                    )
                    with self.profiler.record("pmg_prepare_per_control_step"):
                        self.pmg.prepare(self._obs_history, state)
                    decision_context = self.pmg.decision_context()
                plan_ids, phases = self.env.reference_address()
                reference = self.reference.sample(
                    plan_ids, phases, self.cfg.goal_frames
                )
                goal = self._goal(reference, state)
                if self.pmg is not None:
                    q = reference.now.q[:, : getattr(self.motion, "action_dim", 27)]
                    excess = torch.maximum(
                        torch.relu(
                            q
                            - self.mdp.kin.q_hi[
                                : getattr(self.motion, "action_dim", 27)
                            ]
                        ),
                        torch.relu(
                            self.mdp.kin.q_lo[: getattr(self.motion, "action_dim", 27)]
                            - q
                        ),
                    )
                    generated = plan_ids < 0
                    self._accumulate(
                        {
                            "PMG/reference_joint_limit_frame_fraction": (
                                (excess.gt(1e-4).any(-1) & generated).sum()
                                / generated.sum().clamp_min(1)
                            ),
                            "PMG/reference_joint_limit_max_rad": torch.where(
                                generated[:, None], excess, 0.0
                            ).amax(),
                        }
                    )
                with torch.no_grad():
                    self._update_running_norms(
                        (self.policy.obs_norm, self.policy.goal_norm),
                        (policy_obs, goal),
                    )
                    latent = None
                    if self.perception is not None:
                        perception_frame = self.env.perception()
                        latent = (
                            self.policy.encode_perception(perception_frame["depth"])
                            if self.recurrent
                            else self._encode_perception(perception_frame)
                        )
                        self._update_running_norms(
                            (self.policy.perception_norm,), (latent,)
                        )
                    policy_input = (
                        perception_frame["depth"] if self.recurrent else latent
                    )
                    if self.cfg.actor_relative_orientation:
                        observation = getattr(
                            self.env, "policy_relative_orientation", None
                        )
                        if callable(observation):
                            orientation = observation()
                            if self.pmg is not None:
                                delta = (
                                    self._relative_orientation(state, reference)
                                    - orientation_before_replan
                                )
                                orientation = torch.where(
                                    self.pmg.enabled[:, None],
                                    orientation + delta,
                                    orientation,
                                )
                        else:
                            orientation = self._relative_orientation(
                                state, reference, noise=True
                            )
                        action = self.policy(
                            self._obs_history, goal, policy_input, orientation
                        )
                    else:
                        action = self.policy(self._obs_history, goal, policy_input)
                    noisy_action = action + self.cfg.action_noise * torch.randn_like(
                        action
                    )
                    # Noise precedes the shared execution bound; Policy.forward stays raw.
                    action = clip_action(noisy_action)
                    self._accumulate(
                        {
                            "Collection/action_clip_fraction": (
                                noisy_action.abs() > _COLLECTION_ACTION_LIMIT
                            )
                            .float()
                            .mean(),
                            "Collection/raw_action_abs_max_mean": noisy_action.abs().amax(),
                        }
                    )
                    if self.pmg is not None:
                        plan_ids = plan_ids.clone()
                        self.reference.retain(plan_ids)
                    try:
                        executed = self.env.step(action)
                        self._check_physics()
                        if self.pmg is not None:
                            executed["policy_obs"] = self._obs_history[:, -1]
                            executed.update(decision_context)
                        self.buffer.add(executed)
                    finally:
                        if self.pmg is not None:
                            self.reference.release(plan_ids)
                            self.pmg.release_decision_context()
                    state0 = self._as_state(executed["s"])
                    state1 = self._as_state(executed["s1"])
                    latent1 = None
                    if self.recurrent:
                        latent = self._encode_clean_perception(
                            state0,
                            executed["scan"],
                            executed["hit"],
                            executed["clean_depth"],
                        )
                    elif self.perception is not None:
                        # step() may already have reset the environment. Only
                        # its explicit endpoint belongs to the physical s1.
                        latent1 = self._encode_perception({"depth": executed["depth1"]})
                    self._update_running_norms(
                        (self.wm.in_norm, self.wm.out_norm),
                        (
                            self.wm.inputs(
                                state0,
                                executed["a"],
                                executed["a_prev"],
                                self.normalize_xi(executed["xi"]),
                                latent,
                            ),
                            self.wm.targets(state0, state1)
                            if self.recurrent
                            else self.wm.targets(state0, state1, latent, latent1),
                        ),
                    )
                self.sim_steps += self.cfg.num_envs
                self.transitions += self.cfg.num_envs
                self.global_sim_steps += self.cfg.num_envs * self.world_size
                self.global_transitions += self.cfg.num_envs * self.world_size
                self._accumulate(
                    {
                        "Train/reward": self._mean_tensor(executed["r"]) / 0.1,
                        "Tracking/survival": (
                            1.0 - executed.get("done", executed["term"]).float()
                        ).mean(),
                        "Train/termination": executed["term"].float().mean(),
                        "Tracking/root_height_error_m": (
                            reference.now.p[:, 2] - state0[0][:, 2]
                        )
                        .abs()
                        .mean(),
                        "Tracking/gravity_z_error": (
                            reference.now.R[:, 2, 2] - state0[1][:, 2, 2]
                        )
                        .abs()
                        .mean(),
                    }
                )
                if "ep_len" in executed:
                    self._accumulate(
                        {"Train/episode_length": executed["ep_len"]},
                        sample_counts={
                            "Train/episode_length": executed["ep_len"].numel()
                        },
                    )
                if "ep_ret_norm" in executed:
                    self._accumulate(
                        {
                            "Train/reward_fraction_actual_length": executed[
                                "ep_ret_norm"
                            ]
                        },
                        sample_counts={
                            "Train/reward_fraction_actual_length": executed[
                                "ep_ret_norm"
                            ].numel()
                        },
                    )
                if "terms" in executed:
                    for name, value in executed["terms"].items():
                        self._accumulate({f"Reward/{name}": value})
                    for name in (
                        "e_body_pos",
                        "e_body_ori",
                        "e_root_pos",
                        "action_rate",
                        "joint_limit",
                    ):
                        if name in executed["terms"]:
                            self._accumulate(
                                {f"Tracking/{name}": executed["terms"][name]}
                            )
