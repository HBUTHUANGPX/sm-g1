"""Original evaluation rollout functions."""

from __future__ import annotations


from typing import Any


import torch


from mimic_baseline.tasks.supermimic.core.action_bounds import clip_action


from mimic_baseline.tasks.supermimic.core.mdp import flatten_state, unflatten_state


from mimic_baseline.tasks.supermimic.data.motion import CONTROL_DT


from mimic_baseline.tasks.supermimic.data.reference import (
    ReferenceBatch,
    ensure_static_reference,
)


from mimic_baseline.tasks.supermimic.core.rotations import mat_to_quat


from .constants import (
    PUSH_UPRIGHT_THRESHOLD,
    PUSH_ROOT_HEIGHT_TOLERANCE,
    PUSH_FINAL_DWELL_STEPS,
    METRIC_KEYS,
)
from .metrics import _step_metrics, normalized_clip_return
from .reporting import motion_category


def policy_action(
    policy, history, goal, state, reference: ReferenceBatch, perception=None
):
    """Supply clean current orientation only to actors declaring the new ABI."""
    args = (history, goal) if perception is None else (history, goal, perception)
    if getattr(policy, "relative_orientation_dim", 0):
        from mimic_baseline.tasks.supermimic.core.mdp import relative_root_orientation

        orientation = relative_root_orientation(state[1], reference.now.R)
        return policy(*args, relative_orientation=orientation)
    return policy(*args)


def reference_frames(reference: ReferenceBatch):
    """Source-frame identity is constant for a held target, independent of time."""
    if reference.source_frame is None:
        raise ValueError("Rollout archives require source-frame identity")
    return reference.source_frame


def _policy_obs_history(
    mdp: Any,
    state: tuple[torch.Tensor, ...],
    action: torch.Tensor,
    xi: torch.Tensor,
):
    return mdp.obs(state, action, xi)


def evaluate_tracking(
    env: Any,
    motion: Any,
    mdp: Any,
    policy: torch.nn.Module,
    *,
    goal_frames: int,
    hist: int,
    goal_mode: str = "reference",
    use_original_lengths: bool = False,
    suite: str = "tracking",
    no_termination: bool = False,
    push_speed: float | None = None,
    push_step: int = 50,
    max_recovery_steps: int = 250,
    selected_clip_ids: torch.Tensor | None = None,
    push_speeds: torch.Tensor | None = None,
    record_rollouts: bool = False,
    rollout_sink: list[dict[str, Any]] | None = None,
    archive_sink: list[dict[str, torch.Tensor]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all selected trials in one vectorized simulator rollout."""
    if goal_mode not in ("reference", "state_relative"):
        raise ValueError(f"unsupported goal_mode: {goal_mode!r}")
    motion = ensure_static_reference(motion)
    catalog = motion.catalog
    starts = catalog.clip_start.long()
    ends = (
        catalog.original_clip_end if use_original_lengths else catalog.clip_end
    ).long()
    clip_ids = (
        torch.arange(starts.numel(), device=starts.device)
        if selected_clip_ids is None
        else selected_clip_ids.to(starts.device)
    )
    if env.N != clip_ids.numel():
        raise ValueError(
            f"Environment batch {env.N} does not match {clip_ids.numel()} evaluation trials"
        )
    if push_step < 0:
        raise ValueError("push_step must be non-negative")
    trial_starts = starts[clip_ids]
    trial_ends = ends[clip_ids]
    if push_speeds is not None:
        push_speeds = torch.as_tensor(
            push_speeds, dtype=torch.float32, device=trial_starts.device
        )
        if push_speeds.shape != (env.N,):
            raise ValueError(
                f"push_speeds must have shape {(env.N,)}, got {tuple(push_speeds.shape)}"
            )
    elif push_speed is not None:
        push_speeds = torch.full(
            (env.N,), float(push_speed), dtype=torch.float32, device=trial_starts.device
        )

    if suite == "standing_push":
        transition_limit = int(max_recovery_steps)
        if push_speeds is None:
            raise ValueError("standing_push requires per-trial push speeds")
    else:
        transition_limit = int((trial_ends - trial_starts).max().item())
    if transition_limit < 1:
        raise ValueError("evaluation requires at least one transition")

    env_ids = torch.arange(env.N, dtype=torch.long, device=trial_starts.device)
    env.reset(env_ids, frames=trial_starts)
    archive = None
    if archive_sink is not None:
        from mimic_baseline.tasks.supermimic.rollout_archive import RolloutRecorder

        reference = motion.sample(*motion.locate(env.f))
        archive = RolloutRecorder(
            flatten_state(env.state()),
            flatten_state(reference.now.state),
            reference_frames(reference),
            env.ep_step,
            control_dt=CONTROL_DT,
        )
    active = torch.ones((env.N,), dtype=torch.bool, device=trial_starts.device)
    first_termination = torch.zeros_like(active)
    reached_boundary = torch.zeros_like(active)
    steps = torch.zeros((env.N,), dtype=torch.long, device=trial_starts.device)
    rewards = torch.zeros((env.N,), dtype=torch.float32, device=trial_starts.device)

    # Initialize rollout recorder if requested
    rollout_recorder = None
    rollout_data_raw = None
    if record_rollouts:
        # Store rollout states in list for later processing
        rollout_data_raw = {
            "root_pos": [],
            "root_quat": [],
            "root_linvel": [],
            "root_angvel": [],
            "joint_pos": [],
            "joint_vel": [],
        }

    metric_sums = {
        key: torch.zeros((env.N,), dtype=torch.float32, device=trial_starts.device)
        for key in METRIC_KEYS
    }
    metric_count = torch.zeros_like(rewards)
    final_dwell = torch.zeros_like(steps)
    history: torch.Tensor | None = None
    policy_observation = getattr(env, "policy_observation", None)

    with torch.no_grad():
        for step_index in range(transition_limit):
            state = env.state()
            obs = (
                policy_observation()
                if callable(policy_observation)
                else _policy_obs_history(mdp, state, env.a_prev, env.xi)
            )
            if history is None:
                history = obs.unsqueeze(1).expand(-1, hist, -1).clone()
            else:
                history = torch.cat((history[:, 1:], obs.unsqueeze(1)), 1)
            reference = motion.sample(*motion.locate(env.f), goal_frames=goal_frames)
            goal = reference.goal(mode=goal_mode, rotation=state[1])
            raw_action = policy_action(policy, history, goal, state, reference)
            if not torch.isfinite(raw_action[active]).all():
                raise FloatingPointError(
                    "Nonfinite policy action during rollout export"
                )
            action = clip_action(raw_action)
            action = action.masked_fill(~active.unsqueeze(-1), 0.0)

            if (
                suite == "standing_push"
                and push_speeds is not None
                and step_index == push_step
            ):
                velocity = torch.zeros(
                    (env.N, 6), dtype=torch.float32, device=trial_starts.device
                )
                velocity[:, 0] = torch.where(
                    active, push_speeds, torch.zeros_like(push_speeds)
                )
                env.write_root_link_velocity(velocity, env_ids.to(dtype=torch.int32))

            executed = env.step(action)
            if (
                not torch.isfinite(executed["s1"][active]).all()
                or not torch.isfinite(executed["r"][active]).all()
            ):
                raise FloatingPointError(
                    "Nonfinite simulation state/reward during evaluation"
                )
            if getattr(env, "evaluation_profile", None) == "full_clean":
                if executed["done"].any() or not torch.all(
                    env.ep_step == step_index + 1
                ):
                    raise RuntimeError(
                        "Clean evaluation terminated, truncated or autoreset"
                    )
            reference = motion.sample(*motion.locate(env.f))
            if archive is not None:
                archive.append(
                    executed["s1"],
                    flatten_state(reference.now.state),
                    executed["a"],
                    reference_frames(reference),
                    env.ep_step,
                    executed["term"],
                )
            was_active = active
            active_float = was_active.to(dtype=torch.float32)
            rewards.add_(torch.where(was_active, executed["r"], 0.0))
            steps.add_(was_active.long())

            # Record rollout state if requested
            if record_rollouts and rollout_data_raw is not None:
                current_state = unflatten_state(executed["s1"])
                root_pos, root_R, root_v, root_w, q, qd = current_state
                # Stay on device; a single batched copy happens after the rollout.
                rollout_data_raw["root_pos"].append(root_pos)
                rollout_data_raw["root_quat"].append(mat_to_quat(root_R))
                rollout_data_raw["root_linvel"].append(root_v)
                rollout_data_raw["root_angvel"].append(root_w)
                rollout_data_raw["joint_pos"].append(q)
                rollout_data_raw["joint_vel"].append(qd)

            metrics = _step_metrics(
                executed,
                reference,
                executed.get("terms", {}),
            )
            for key in METRIC_KEYS:
                metric_sums[key].add_(torch.where(was_active, metrics[key], 0.0))
            metric_count.add_(active_float)

            term_now = executed["term"].bool() & was_active
            first_termination.logical_or_(term_now)
            if suite == "tracking":
                boundary = (
                    steps >= trial_ends - trial_starts
                    if use_original_lengths
                    else reference.expiry.episode_end
                )
            else:
                boundary = steps >= max_recovery_steps
                actual = unflatten_state(executed["s1"])
                upright = actual[1][..., 2, 2] >= PUSH_UPRIGHT_THRESHOLD
                height_ok = (
                    actual[0][..., 2] - reference.now.p[..., 2]
                ).abs() <= PUSH_ROOT_HEIGHT_TOLERANCE
                final_dwell = torch.where(
                    was_active & upright & height_ok & (step_index >= push_step),
                    final_dwell + 1,
                    torch.zeros_like(final_dwell),
                )
            boundary &= was_active
            reached_boundary.logical_or_(boundary)
            stopping = boundary if no_termination else (boundary | term_now)
            active = active & ~stopping
            # Check once per 32 steps, avoiding a device sync at every tick.
            if (step_index + 1) % 32 == 0 and not active.any():
                break

    reported_speeds = (
        push_speeds
        if push_speeds is not None
        else torch.zeros((env.N,), dtype=torch.float32, device=trial_starts.device)
    )
    packed = torch.cat(
        (
            rewards[:, None],
            steps.float()[:, None],
            first_termination.float()[:, None],
            reached_boundary.float()[:, None],
            final_dwell.float()[:, None],
            torch.stack([metric_sums[key] for key in METRIC_KEYS], dim=1),
            metric_count[:, None],
            trial_starts.float()[:, None],
            trial_ends.float()[:, None],
            reported_speeds[:, None],
            clip_ids.float()[:, None],
        ),
        dim=1,
    ).cpu()
    host = packed.tolist()

    records: list[dict[str, Any]] = []
    metric_offset = 5
    count_offset = metric_offset + len(METRIC_KEYS)
    for index, row in enumerate(host):
        (
            reward,
            step_value,
            terminated_value,
            boundary_value,
            dwell_value,
        ) = row[:metric_offset]
        metric_values = {
            key: row[metric_offset + metric_index]
            for metric_index, key in enumerate(METRIC_KEYS)
        }
        count = row[count_offset]
        start = int(row[count_offset + 1])
        end = int(row[count_offset + 2])
        push_speed_value = float(row[count_offset + 3])
        clip_id = int(row[-1])
        clip_steps = int(step_value)
        terminated = bool(terminated_value)
        reached_end = bool(boundary_value)
        # With termination disabled the hard-termination flag is diagnostic
        # only, so completion is decided by reaching the clip end. Physical
        # success is judged separately from the recorded rollout.
        completed = reached_end if no_termination else (reached_end and not terminated)
        if terminated:
            outcome = (
                "diagnostic_hard_termination_observed"
                if no_termination
                else "hard_termination"
            )
        elif reached_end:
            outcome = "clip_end" if suite == "tracking" else "diagnostic_complete"
        else:
            outcome = "incomplete"
        name = str(catalog.names[clip_id])
        record = {
            "suite": suite,
            "name": name,
            "category": motion_category(name),
            "clip_id": clip_id,
            "clip_start": start,
            "clip_end": end,
            "full_clip_frames": end + 1 - start,
            "steps": clip_steps,
            "return": float(reward),
            "normalized_return": normalized_clip_return(
                float(reward),
                start,
                end,
                duration_steps=clip_steps if suite == "standing_push" else None,
            ),
            "terminated": terminated,
            "completed": completed,
            "covered_to_end": reached_end,
            "tracking_success": None if no_termination else completed,
            "outcome": outcome,
            "termination_policy": "disabled_diagnostic"
            if no_termination
            else "native_hard_termination",
            "push_speed": push_speed_value if push_speeds is not None else None,
            "diagnostic_final_dwell_ok": (
                suite == "standing_push"
                and int(dwell_value) >= PUSH_FINAL_DWELL_STEPS
                and reached_end
            ),
        }
        for key, value in metric_values.items():
            record[key] = float((value / max(count, 1.0)) ** 0.5)
        records.append(record)

    # Finalize rollout recordings if requested: one batched host copy, not per step.
    rollout_data = None
    if record_rollouts and rollout_data_raw is not None:
        import numpy as np

        host_steps = steps.cpu().tolist()
        stacked = {
            key: torch.stack(value, dim=1).cpu().numpy()
            for key, value in rollout_data_raw.items()
        }
        rollout_data = {}
        for env_idx in range(env.N):
            n_steps = int(host_steps[env_idx])
            rollout_data[f"env_{env_idx}"] = {
                key: value[env_idx, :n_steps] for key, value in stacked.items()
            }
            rollout_data[f"env_{env_idx}"]["num_steps"] = n_steps
        if rollout_sink is not None:
            rollout_sink.append(rollout_data)

    if archive is not None:
        archive_sink.extend(archive.finish([record["steps"] for record in records]))
    return records
