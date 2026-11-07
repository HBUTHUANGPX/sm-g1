"""Explicit evaluation protocols; training configurations are never mutated."""

from __future__ import annotations

PROFILES = ("strict", "full_clean")
PROTOCOL_VERSION = 2


def configure_evaluation(cfg, profile, *, longest_clip_steps):
    from ..data.motion import CONTROL_DT

    if profile not in PROFILES or not cfg.eval_mode:
        raise ValueError("An evaluation profile requires eval_mode=True")
    if longest_clip_steps < 1:
        raise ValueError("Evaluation requires a nonempty trajectory")
    cfg.reset_noise = False
    # Keep the timeout term enabled in strict mode, beyond every original endpoint.
    cfg.episode_length_s = (int(longest_clip_steps) + 2) * CONTROL_DT
    if profile == "full_clean":
        # Per-reset random actuator lag is also randomization.
        cfg.min_delay = cfg.max_delay = 0
        # Config post-init has already materialized the robot actuator configs.
        for actuator in cfg.scene.robot.actuators.values():
            if hasattr(actuator, "max_delay"):
                actuator.min_delay = actuator.max_delay = 0


def runtime_profile(env, profile):
    """Check realized managers and reset state, not merely requested CLI flags."""
    import torch

    cfg = env.cfg
    command = env.command_manager.get_term("motion")
    groups = {
        name: bool(group.enable_corruption)
        for name, group in vars(cfg.observations).items()
        if hasattr(group, "enable_corruption") and not isinstance(group, type)
    }
    if (
        not cfg.eval_mode
        or not env.randomization.eval_mode
        or command.reset_noise
        or any(groups.values())
    ):
        raise RuntimeError("Evaluation randomization/noise is still enabled")
    if not torch.all(env.xi[:, 0] == 1) or not torch.all(env.xi[:, 1:] == 0):
        raise RuntimeError("Evaluation physics context is not nominal")
    if torch.any(env.recovery_role):
        raise RuntimeError(
            "Random recovery reset roles must be disabled during evaluation"
        )
    terms = list(env.termination_manager.active_terms)
    delays = {}
    for name, actuator in env.scene["robot"].actuators.items():
        buffer = getattr(actuator, "positions_delay_buffer", None)
        if buffer is not None:
            delays[name] = sorted(set(buffer.time_lags.detach().cpu().tolist()))
    if profile == "full_clean":
        if (
            terms
            or cfg.max_delay != 0
            or any(lag != 0 for lags in delays.values() for lag in lags)
        ):
            raise RuntimeError(
                "Clean evaluation still has termination or actuator delay"
            )
    elif not {"height", "tilt", "motion_end", "time_out"}.issubset(terms):
        raise RuntimeError("Strict evaluation is missing native termination terms")
    if (
        profile == "strict"
        and cfg.recovery_enabled
        and not {"ankle", "wrist"}.issubset(terms)
    ):
        raise RuntimeError("Strict recovery evaluation is missing tracking groups")
    return {
        "profile": profile,
        "protocol_version": PROTOCOL_VERSION,
        "domain_randomization": False,
        "reset_noise": False,
        "observation_corruption": groups,
        "active_termination_terms": terms,
        "action_delay_ticks": delays,
        "recovery_reset_roles": False,
        "termination_timeout_seconds": cfg.episode_length_s,
        "full_trajectory_is_tracking_success": False,
    }
