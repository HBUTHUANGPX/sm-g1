from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mimic_baseline.tasks.supermimic.data.lafan1 import convert_clip
from mimic_baseline.tasks.supermimic.data.motion import MotionLib
from mimic_baseline.tasks.supermimic.data.reference import StaticReference
from mimic_baseline.tasks.supermimic.robots.g1.kinematics import G1Kinematics
from mimic_baseline.tasks.supermimic.core.mdp import flatten_state
from mimic_baseline.tasks.supermimic.evaluation.rollout import evaluate_tracking
from test_kinematics import make_urdf


def make_motion(tmp_path):
    paths = []
    for i, frames in enumerate((4, 7)):
        raw = np.zeros((frames, 36))
        raw[:, 6] = 1
        source = tmp_path / f"walk{i}.csv"
        path = source.with_suffix(".npz")
        np.savetxt(source, raw, delimiter=",")
        convert_clip(source, path)
        paths.append(path)
    return MotionLib(paths, G1Kinematics(make_urdf(tmp_path), "cpu"), "cpu")


class FollowingEnv:
    def __init__(self, motion, fail_at=None):
        self.motion = motion
        self.N = 2
        self.f = motion.clip_start.clone()
        self.ep_step = torch.zeros(2, dtype=torch.long)
        self.a_prev = torch.zeros(2, 29)
        self.xi = torch.zeros(2, 33)
        self.fail_at = fail_at
        self.resets = 0

    def reset(self, env_ids, frames):
        self.resets += 1
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0

    def state(self):
        return (
            StaticReference(self.motion)
            .sample(*StaticReference(self.motion).locate(self.f))
            .now.state
        )

    def policy_observation(self):
        return torch.zeros(2, 93)

    def step(self, action):
        self.ep_step += 1
        self.f = torch.minimum(self.f + 1, self.motion.clip_end)
        term = (
            torch.zeros(2, dtype=torch.bool)
            if self.fail_at is None
            else self.ep_step >= self.fail_at
        )
        return {
            "s1": flatten_state(self.state()),
            "r": torch.full((2,), 0.01),
            "term": term,
            "terms": {"e_body_pos": torch.zeros(2), "e_body_ori": torch.zeros(2)},
        }


@pytest.mark.parametrize("no_termination", [False, True])
def test_each_clip_stops_at_original_unpadded_endpoint(tmp_path, no_termination):
    motion = make_motion(tmp_path)
    env = FollowingEnv(motion)
    assert torch.any(motion.clip_len > motion.original_clip_len)
    rows = evaluate_tracking(
        env,
        motion,
        None,
        lambda h, g: torch.zeros(2, 29),
        goal_frames=1,
        hist=1,
        use_original_lengths=True,
        no_termination=no_termination,
    )
    assert [r["steps"] for r in rows] == (motion.original_clip_len - 1).tolist()
    assert all(r["completed"] for r in rows)
    assert env.resets == 1


def test_strict_stops_on_failure_but_diagnostic_rollout_continues(tmp_path):
    motion = make_motion(tmp_path)

    def run(ignore):
        env = FollowingEnv(motion, fail_at=2)
        return evaluate_tracking(
            env,
            motion,
            None,
            lambda h, g: torch.zeros(2, 29),
            goal_frames=1,
            hist=1,
            use_original_lengths=True,
            no_termination=ignore,
        )

    strict = run(False)
    full = run(True)
    assert [r["steps"] for r in strict] == [2, 2] and not any(
        r["completed"] for r in strict
    )
    assert [r["steps"] for r in full] == (motion.original_clip_len - 1).tolist()
    assert all(r["covered_to_end"] and r["tracking_success"] is None for r in full)


def test_clean_profile_requires_evaluation_and_disables_delay_noise():
    from mimic_baseline.tasks.supermimic.evaluation.profiles import configure_evaluation

    actuator = SimpleNamespace(min_delay=0, max_delay=4)
    cfg = SimpleNamespace(
        scene=SimpleNamespace(robot=SimpleNamespace(actuators={"joints": actuator})),
        eval_mode=True,
        max_delay=4,
        min_delay=0,
        reset_noise=True,
        episode_length_s=10,
    )
    configure_evaluation(cfg, "full_clean", longest_clip_steps=1000)
    assert cfg.max_delay == cfg.min_delay == 0 and cfg.reset_noise is False
    assert actuator.max_delay == actuator.min_delay == 0
    assert cfg.episode_length_s > 20
    with pytest.raises(ValueError):
        configure_evaluation(
            SimpleNamespace(eval_mode=False), "full_clean", longest_clip_steps=1
        )


def test_clean_runtime_audit_rejects_realized_delay_or_termination():
    from mimic_baseline.tasks.supermimic.evaluation.profiles import runtime_profile

    delay = SimpleNamespace(time_lags=torch.zeros(2, dtype=torch.long))
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            eval_mode=True,
            observations=SimpleNamespace(obs=SimpleNamespace(enable_corruption=False)),
            max_delay=0,
            episode_length_s=100,
            recovery_enabled=True,
        ),
        command_manager=SimpleNamespace(
            get_term=lambda _: SimpleNamespace(reset_noise=False)
        ),
        randomization=SimpleNamespace(eval_mode=True),
        xi=torch.zeros(2, 33),
        recovery_role=torch.zeros(2, dtype=torch.bool),
        termination_manager=SimpleNamespace(active_terms=[]),
        scene={
            "robot": SimpleNamespace(
                actuators={"joints": SimpleNamespace(positions_delay_buffer=delay)}
            )
        },
    )
    env.xi[:, 0] = 1
    assert runtime_profile(env, "full_clean")["action_delay_ticks"] == {"joints": [0]}
    delay.time_lags[0] = 1
    with pytest.raises(RuntimeError, match="delay"):
        runtime_profile(env, "full_clean")
    delay.time_lags.zero_()
    env.termination_manager.active_terms = ["time_out"]
    with pytest.raises(RuntimeError, match="termination"):
        runtime_profile(env, "full_clean")


@pytest.mark.parametrize("fault", ["done", "autoreset", "nan"])
def test_clean_rollout_detects_runtime_faults(tmp_path, fault):
    motion = make_motion(tmp_path)
    env = FollowingEnv(motion)
    env.evaluation_profile = "full_clean"
    original = env.step

    def step(action):
        result = original(action)
        result["done"] = torch.zeros(2, dtype=torch.bool)
        if fault == "done":
            result["done"][0] = True
        if fault == "autoreset":
            env.ep_step.zero_()
        if fault == "nan":
            result["s1"][0, 0] = float("nan")
        return result

    env.step = step
    with pytest.raises((RuntimeError, FloatingPointError)):
        evaluate_tracking(
            env,
            motion,
            None,
            lambda h, g: torch.zeros(2, 29),
            goal_frames=1,
            hist=1,
            use_original_lengths=True,
            no_termination=True,
        )


def test_clean_reports_do_not_label_coverage_as_tracking_completion(tmp_path):
    import json
    from mimic_baseline.tasks.supermimic.evaluation.clean_reporting import (
        write_clean_results,
    )

    grouped = write_clean_results(
        tmp_path,
        [
            dict(
                name="walk",
                category="walk",
                completed=True,
                covered_to_end=True,
                tracking_success=None,
                normalized_return=0.1,
            )
        ],
    )
    assert (
        grouped["walk"]["coverage_rate"] == 1
        and grouped["walk"]["tracking_success"] is None
    )
    assert "completion_rate" not in grouped["walk"]
    assert (
        "completed"
        not in json.loads((tmp_path / "perclip.json").read_text())["clips"][0]
    )
    assert "completed" not in (tmp_path / "perclip.csv").read_text()
