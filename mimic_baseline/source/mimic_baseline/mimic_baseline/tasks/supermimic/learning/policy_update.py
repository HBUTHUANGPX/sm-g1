from __future__ import annotations


from functools import partial


import torch


import torch.nn as nn


from torch.utils.checkpoint import checkpoint


from mimic_baseline.tasks.supermimic.core.action_bounds import (
    ACTION_LIMIT as _COLLECTION_ACTION_LIMIT,
    clip_action,
)


from mimic_baseline.tasks.supermimic.core.mdp import MAX_STEP_REWARD


from mimic_baseline.tasks.supermimic.learning.perception import height_scan_orientation


from mimic_baseline.tasks.supermimic.core.recovery import remaining_protection


from mimic_baseline.tasks.supermimic.data.reference import ReferenceBatch


from .common import _valid_depth


class PolicyUpdateMixin:
    """Original Trainer methods; state is owned by Trainer."""

    def _initial_observation_history(self, history):
        """Reconstruct the actor's sensor history from sampled replay states."""
        with torch.no_grad():
            observation = self._obs(
                self._as_state(history["s"]), history["a_prev"], history["xi"]
            )
        noisy = self.mdp.add_noise(observation)
        # Replay clamps missing predecessors to the oldest retained frame.
        # Reuse that frame's noisy observation, including when it is the seed,
        # just as collection repeats one already-noisy observation on reset.
        # Keep the original draw shape/RNG budget; real history slots retain
        # their exact draws. Episode steps also handle an overwritten ring edge.
        episode_steps = history["ep_step"].long()
        source = noisy.shape[1] - 1 - (episode_steps[:, -1:] - episode_steps)
        return noisy.gather(1, source[..., None].expand_as(noisy))

    def _critic_input(
        self,
        state: tuple[torch.Tensor, ...],
        reference: ReferenceBatch,
        xi_n: torch.Tensor,
        goal: torch.Tensor,
        latent: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        remaining: torch.Tensor | None = None,
        body_state: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        context = (
            {"remaining": remaining, "body_state": body_state}
            if self.cfg.recovery_enabled
            else {}
        )
        parts = [self.mdp.critic_features(state, reference, xi_n, **context), goal]
        if self.recurrent:
            scan, hit = latent
            parts.extend((scan, hit.to(scan.dtype)))
            if self.critic.world_frame:
                parts.append(height_scan_orientation(state[1]))
        elif self.cfg.critic_perception:
            if latent is None or latent.shape[-1] != self.cfg.perception_latent_dim:
                raise ValueError(
                    "critic_perception requires a latent with width "
                    f"{self.cfg.perception_latent_dim}"
                )
            parts.append(latent)
        result = torch.cat(parts, -1)
        expected = getattr(self.critic, "in_dim", result.shape[-1])
        if result.shape[-1] != expected:
            raise ValueError(
                f"critic input width {result.shape[-1]} does not match Critic({expected}); "
                f"critic_perception={self.cfg.critic_perception}"
            )
        return result

    def _wm_modules(self) -> tuple[nn.Module, ...]:
        """Modules sharing the world-model optimizer and its gradient budget."""
        if self.perception is None:
            return (self.wm,)
        return (self.wm, self.perception)

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(value)

    def _bootstrap_value(self, value: torch.Tensor) -> torch.Tensor:
        """Optional A/B bound on targets; raw predictions still train under MSE."""
        if not self.cfg.bound_bootstrap_value:
            return value
        upper = MAX_STEP_REWARD / (1.0 - self.cfg.gamma)
        # A finite reward bound must not hide NaN/Inf in a failed critic.
        return torch.where(torch.isfinite(value), value.clamp_max(upper), value)

    def update_policy_and_critic(self) -> dict[str, torch.Tensor]:
        """One 4096-rollout BPTT actor update followed by five critic epochs."""
        self._check_physics()
        with self.profiler.record("actor"):
            seed, history = self.buffer.sample_starts(
                self.cfg.policy_batch,
                self.cfg.hist,
                self.cfg.horizon,
                self.motion,
                **(
                    {"bin_probabilities": self.sampler.actor_probabilities}
                    if self.sampler is not None
                    else {}
                ),
            )
            state0 = self._as_state(seed["s"])
            xi = seed["xi"]
            xi_n = self.normalize_xi(xi)
            plan_ids, phases0 = seed["ref_id"], seed["ref_phase"]
            reference0 = self.reference.sample(plan_ids, phases0, self.cfg.goal_frames)
            reference = reference0
            action_prev = seed["a_prev"]
            state = state0
            goal0 = self._goal(reference0, state0)
            goal = goal0
            obs_history = self._initial_observation_history(history)
            states = []
            actions = []
            action_clip_counts = []
            action_maxima = []
            previous_actions = []
            goals = []
            # The latent entering imagination is the one encoded from the seed's
            # recorded frame. From there the world model predicts it, so the
            # actor never sees a real depth frame the rollout could not have.
            latent = None
            latents = []
            height_history = []
            frozen_modules = (*self._wm_modules(), self.critic)
            frozen_state = [
                (
                    module.training,
                    [parameter.requires_grad for parameter in module.parameters()],
                )
                for module in frozen_modules
            ]
            for module in frozen_modules:
                module.eval()
                self._set_requires_grad(module, False)
            try:
                policy_forward = self.policy
                model_step = self._clean_prediction if self.recurrent else self.wm.step
                # These calls are deterministic and only read normalization state.
                # Noise, camera queues and geometry remain outside recomputation.
                if self.cfg.actor_recompute != "none":
                    policy_forward = partial(
                        checkpoint,
                        policy_forward,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                if self.cfg.actor_recompute == "full":
                    # The nested depth-decoder checkpoint creates factory tensors.
                    # Its forward and backward recomputation need the same device.
                    context = (
                        {
                            "context_fn": lambda: (
                                torch.device(self.device),
                                torch.device(self.device),
                            )
                        }
                        if self.recurrent
                        else {}
                    )
                    model_step = partial(
                        checkpoint,
                        model_step,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        **context,
                    )
                if self.recurrent:
                    hidden = self._burnin_hidden(seed["burnin"])
                    scan, hit = seed["scan"].float(), seed["hit"]
                    depth = seed["clean_depth"].float()
                    camera_state = self.camera.seed(
                        seed["camera_frames"].float(), seed["camera_lag"].long()
                    )
                    latent = self.camera.read(camera_state)
                    height_history.append((scan, hit))
                    # Imagined episodes do not reset. Capture rows repeat every
                    # camera period; compact five phases once instead of H times.
                    capture_rows = tuple(
                        ((seed["ep_step"] + offset) % self.camera.capture_steps == 0)
                        .nonzero()
                        .flatten()
                        for offset in range(self.camera.capture_steps)
                    )
                elif self.perception is not None:
                    with torch.no_grad():
                        latent = self.perception.encode(
                            seed["depth"], _valid_depth(seed["depth"])
                        )
                if self.cfg.critic_perception and not self.recurrent:
                    latents.append(latent)
                for step in range(self.cfg.horizon):
                    next_reference = self.reference.sample(
                        plan_ids, phases0 + step + 1, self.cfg.goal_frames
                    )
                    previous_actions.append(action_prev)
                    if self.cfg.actor_relative_orientation:
                        # Use this pre-action predicted state/frame. Pass the tensor
                        # explicitly so recomputation cannot read a later reference.
                        orientation = self._relative_orientation(
                            state, reference, noise=True
                        )
                        raw_action = policy_forward(
                            obs_history, goal, latent, orientation
                        )
                    else:
                        raw_action = policy_forward(obs_history, goal, latent)
                    action = clip_action(raw_action)
                    raw_magnitude = raw_action.detach().abs()
                    action_clip_counts.append(
                        (raw_magnitude > _COLLECTION_ACTION_LIMIT).sum()
                    )
                    action_maxima.append(raw_magnitude.amax())
                    if self.recurrent:
                        state, hidden, scan, logits, depth = model_step(
                            state, action, action_prev, xi_n, scan, hit, depth, hidden
                        )
                        hit = logits.sigmoid()
                        camera_state, latent = self.camera.advance(
                            camera_state,
                            depth,
                            seed["ep_step"] + step + 1,
                            capture_ids=capture_rows[
                                (step + 1) % self.camera.capture_steps
                            ],
                            occlusion_query=lambda clean, ids: (
                                self.camera_geometry.visibility(
                                    tuple(value[ids] for value in state),
                                    motion_ids=next_reference.scene_id[ids],
                                )
                            ),
                        )
                        height_history.append((scan, hit))
                    elif self.perception is None:
                        state = model_step(state, action, action_prev, xi_n)
                    else:
                        state, latent = model_step(
                            state, action, action_prev, xi_n, latent
                        )
                    if self.cfg.critic_perception and not self.recurrent:
                        latents.append(latent)
                    reference = next_reference
                    obs = self.mdp.add_noise(self._obs(state, action, xi))
                    obs_history = torch.cat((obs_history[:, 1:], obs.unsqueeze(1)), 1)
                    goal = self._goal(reference, state)
                    states.append(state)
                    actions.append(action)
                    goals.append(goal)
                    action_prev = action

                batch = self.cfg.policy_batch
                horizon = self.cfg.horizon
                flat = lambda values: torch.stack(values).flatten(0, 1)  # noqa: E731
                states_flat = tuple(
                    flat([item[i] for item in states]) for i in range(6)
                )
                future_phases = (
                    phases0[None]
                    + torch.arange(1, horizon + 1, device=self.device)[:, None]
                )
                rollout_ids = plan_ids[None].expand_as(future_phases)
                rollout_reference = self.reference.sample(
                    rollout_ids.flatten(), future_phases.flatten(), bodies=True
                )
                actions_flat = flat(actions)
                previous_flat = flat(previous_actions)
                body_context, clock_context, tail_context = {}, {}, {}
                if self.cfg.recovery_enabled:
                    # Share the post-state FK across reward, hazard and both
                    # critic paths. The WM never predicts an episode clock.
                    bodies = self.mdp.kin.forward(*states_flat)
                    body_context = {"body_state": bodies}
                    ages = (
                        seed["ep_step"][None]
                        + torch.arange(horizon, device=self.device)[:, None]
                    )
                    roles = seed["recovery_role"][None].expand_as(ages)
                    clock_context = {
                        "recovery_role": roles.flatten(),
                        "ep_step": ages.flatten(),
                    }
                    tail_remaining = remaining_protection(
                        seed["recovery_role"], seed["ep_step"] + horizon
                    )
                    tail_context = {
                        "remaining": tail_remaining,
                        "body_state": tuple(
                            value.reshape(horizon, batch, *value.shape[1:])[-1]
                            for value in bodies
                        ),
                    }
                rewards = self.mdp.reward(
                    states_flat,
                    rollout_reference,
                    actions_flat,
                    previous_flat,
                    **body_context,
                ).view(horizon, batch)
                termination = self.mdp.soft_termination(
                    states_flat, rollout_reference, **body_context, **clock_context
                ).view(horizon, batch)
                source_reference = self.reference.sample(rollout_ids, future_phases - 1)
                valid_reference = reference0.valid[None] & source_reference.valid
                clip_finished = rollout_reference.expiry.episode_end.view(
                    horizon, batch
                )
                rewards = torch.where(valid_reference, rewards, 0.0)
                termination = torch.where(clip_finished, 1.0, termination)
                alive = torch.cumprod(1.0 - termination, 0)
                survival_at_step = torch.cat(
                    (torch.ones(1, batch, device=self.device), alive[:-1]), 0
                )
                discount = self.cfg.gamma ** torch.arange(
                    horizon, device=self.device, dtype=torch.float32
                ).unsqueeze(-1)
                imagined_return = (discount * survival_at_step * rewards).sum(0)
                survival = alive[-1]
                tail_reference = self.reference.sample(
                    plan_ids, phases0 + horizon, bodies=True
                )
                final_critic = self.critic(
                    self._critic_input(
                        state,
                        tail_reference,
                        xi_n,
                        goal,
                        (scan, hit) if self.recurrent else latent,
                        **tail_context,
                    )
                )
                reward_return = imagined_return
                bootstrap_return = (
                    (self.cfg.gamma**horizon)
                    * survival
                    * self._bootstrap_value(final_critic)
                )
                imagined_return = reward_return + bootstrap_return
                replay_weight = (
                    seed["sampling_weight"] if self.sampler is not None else 1.0
                )
                actor_loss = -(imagined_return * replay_weight).mean()
                self.opt_policy.zero_grad(set_to_none=True)
                actor_loss.backward()
                self._average_gradients(self.policy)
                actor_grad = self._clip_gradient_norm("actor update", self.policy)
                nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
                self.opt_policy.step()
                self.optimizer_steps += 1

                reward_sum = (survival_at_step * rewards).detach().sum(0)
                tail_over_bound = (
                    (
                        (
                            final_critic.detach()
                            > MAX_STEP_REWARD / (1.0 - self.cfg.gamma)
                        )
                        & torch.isfinite(final_critic.detach())
                    )
                    .float()
                    .mean()
                )
                actor_metrics = {
                    "Loss/actor": actor_loss.detach(),
                    "Gradient/actor": actor_grad.detach(),
                    "Tracking/imagined_return": (reward_sum / horizon / 0.1)
                    .mean()
                    .detach(),
                    "Tracking/imagined_survival": survival.mean().detach(),
                    "Actor/reward_return": (reward_return * replay_weight)
                    .mean()
                    .detach(),
                    "Actor/bootstrap_return": (bootstrap_return * replay_weight)
                    .mean()
                    .detach(),
                    "Critic/raw_tail_value_max_mean": final_critic.detach().amax(),
                    "Critic/raw_tail_over_bound_fraction": tail_over_bound,
                    "Critic/bootstrap_bound_fraction": (
                        tail_over_bound
                        if self.cfg.bound_bootstrap_value
                        else final_critic.new_zeros(())
                    ),
                    "Imagination/action_clip_fraction": (
                        torch.stack(action_clip_counts).sum()
                        / (horizon * batch * actions[0].shape[-1])
                    ),
                    "Imagination/raw_action_abs_max_mean": torch.stack(
                        action_maxima
                    ).amax(),
                }
            finally:
                for module, (training, gradients) in zip(
                    frozen_modules, frozen_state, strict=True
                ):
                    module.train(training)
                    for parameter, requires_grad in zip(
                        module.parameters(), gradients, strict=True
                    ):
                        parameter.requires_grad_(requires_grad)

        with self.profiler.record("critic"):
            with torch.no_grad():
                states_with_seed = tuple(
                    torch.cat(
                        (
                            state0[i].unsqueeze(0),
                            torch.stack([item[i].detach() for item in states]),
                        ),
                        0,
                    )
                    for i in range(6)
                )
                phases_with_seed = (
                    phases0[None]
                    + torch.arange(horizon + 1, device=self.device)[:, None]
                )
                ids_with_seed = plan_ids[None].expand_as(phases_with_seed)
                critic_reference = self.reference.sample(
                    ids_with_seed.flatten(), phases_with_seed.flatten(), bodies=True
                )
                goals_with_seed = torch.cat((goal0.unsqueeze(0), torch.stack(goals)), 0)
                xi_with_seed = xi_n.unsqueeze(0).expand(self.cfg.horizon + 1, -1, -1)
                flat_states = tuple(value.flatten(0, 1) for value in states_with_seed)
                critic_perception = (
                    tuple(
                        torch.stack([item[i] for item in height_history]).flatten(0, 1)
                        for i in range(2)
                    )
                    if self.recurrent
                    else torch.stack(latents).flatten(0, 1)
                    if latents
                    else None
                )
                critic_context = {}
                if self.cfg.recovery_enabled:
                    state_ages = (
                        seed["ep_step"][None]
                        + torch.arange(horizon + 1, device=self.device)[:, None]
                    )
                    remaining = remaining_protection(
                        seed["recovery_role"][None], state_ages
                    )
                    seed_bodies = self.mdp.kin.forward(*state0)
                    critic_bodies = tuple(
                        torch.cat(
                            (
                                start[None],
                                future.detach().reshape(
                                    horizon, batch, *future.shape[1:]
                                ),
                            ),
                            0,
                        ).flatten(0, 1)
                        for start, future in zip(seed_bodies, bodies, strict=True)
                    )
                    critic_context = {
                        "remaining": remaining.flatten(),
                        "body_state": critic_bodies,
                    }
                critic_x = self._critic_input(
                    flat_states,
                    critic_reference,
                    xi_with_seed.flatten(0, 1),
                    goals_with_seed.flatten(0, 1),
                    critic_perception,
                    **critic_context,
                )
                critic_x = critic_x.view(
                    self.cfg.horizon + 1, self.cfg.policy_batch, -1
                )
                norm_input = critic_x.flatten(0, 1)
                if self.recurrent:
                    norm_input = self.critic.normalization_features(norm_input)
                valid_states = torch.cat((reference0.valid[None], valid_reference), 0)
                self._update_running_norms(
                    (self.critic.norm,), (norm_input[valid_states.flatten()],)
                )
                values = self._bootstrap_value(self.critic(critic_x))
                returns = values[-1]
                targets_descending = []
                for step in range(self.cfg.horizon - 1, -1, -1):
                    returns = rewards[step].detach() + self.cfg.gamma * (
                        1.0 - termination[step].detach()
                    ) * (
                        (1.0 - self.cfg.gae_lambda) * values[step + 1]
                        + self.cfg.gae_lambda * returns
                    )
                    targets_descending.append(returns)
                targets = torch.stack(targets_descending[::-1])
                critic_inputs = critic_x[:-1].reshape(-1, critic_x.shape[-1])
                target_values = targets.reshape(-1)
                valid_critic = valid_reference.reshape(-1)
                critic_weight = (
                    torch.ones_like(valid_reference, dtype=targets.dtype)
                    * replay_weight
                ).reshape(-1)

            count = critic_inputs.shape[0]
            minibatch = max(1, count // self.cfg.critic_minibatches)
            critic_loss = torch.zeros((), device=self.device)
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(count, device=self.device)
                for index in range(self.cfg.critic_minibatches):
                    selected = permutation[index * minibatch : (index + 1) * minibatch]
                    if selected.numel() == 0:
                        continue
                    prediction = self.critic(critic_inputs[selected])
                    # Paper Eq. (3): time SUM / batch MEAN. The reference code
                    # instead takes a time-and-batch mean (a factor H smaller).
                    squared_error = (prediction - target_values[selected]).square()
                    critic_loss = (
                        self.cfg.horizon
                        * torch.where(
                            valid_critic[selected],
                            squared_error * critic_weight[selected],
                            0.0,
                        ).mean()
                    )
                    self.opt_critic.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    self._average_gradients(self.critic)
                    critic_grad = self._clip_gradient_norm("critic update", self.critic)
                    self.opt_critic.step()
                    self.optimizer_steps += 1

            actor_metrics.update(
                {
                    "Loss/critic": critic_loss.detach(),
                    "Gradient/critic": critic_grad.detach(),
                }
            )
        return actor_metrics
