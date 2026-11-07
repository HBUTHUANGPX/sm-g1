"""Read-only reset metrics for mastery-weighted motion sampling."""


def update_motion_bin_returns(env, env_ids, command_name="motion"):
    # Statistics are updated before reset on every rank, never in this callback.
    cmd = env.command_manager.get_term(command_name)
    return {"mean_bin_mastery": cmd.mean_bin_mastery(), "reset_frame_mean": cmd.mean_reset_frame()}
