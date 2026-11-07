"""Real GPU assertions for the original manager environment, not a simulator double."""

import torch
from .mdp_legacy import flatten_state, unflatten_state
from .rotations import quat_to_mat


@torch.no_grad()
def check_environment(env):
    from isaaclab.envs import ManagerBasedRLEnv

    assert isinstance(env.env, ManagerBasedRLEnv)
    assert env.action_manager.total_action_dim == 29
    ids = torch.arange(env.N, device=env.device)
    frames = env.motion.clip_start[ids % len(env.motion.names)]
    env.reset(ids, frames=frames)
    assert env.policy_observation().shape == (env.N, 93)
    kin = env.mdp.kin
    robot = env.scene["robot"]
    bp, bR, _, _ = kin.forward(*env.state())
    body_ids = [robot.body_names.index(n) for n in kin.tracked_body_names]
    poses = robot.data.body_link_pose_w.torch[:, body_ids]
    sim_R = quat_to_mat(poses[..., [6, 3, 4, 5]])
    pos_error = (bp - (poses[..., :3] - env.scene.env_origins[:, None])).norm(dim=-1).max().item()
    rot_error = (bR - sim_R).abs().max().item()
    assert pos_error < 1e-3 and rot_error < 1e-3, (pos_error, rot_error)
    randomization = env.env.randomization
    context_before = env.env.xi.clone()
    com_before = robot.data.body_com_pose_b.torch.clone()
    before = flatten_state(env.state()).clone()
    env.reset(ids[:1], frames=frames[:1])
    partial_error = (flatten_state(env.state())[1:] - before[1:]).abs().max().item() if env.N > 1 else 0.0
    assert partial_error < 1e-5, partial_error
    assert torch.equal(env.env.xi[1:], context_before[1:])
    assert torch.equal(robot.data.body_com_pose_b.torch[1:], com_before[1:])
    com_expected = randomization._nominal_com[:, :3] + env.env.xi[:, 30:33]
    com_error = (robot.data.body_com_pose_b.torch[:, randomization.torso_id, :3] - com_expected).abs().max().item()
    assert com_error < 1e-6, com_error
    friction_error = None
    if randomization._newton_manager is not None:
        friction_error = (randomization._newton_mu - env.env.xi[:, :1]).abs().max().item()
        assert friction_error < 1e-6, friction_error
    terminal_count = 0
    reward_error = 0.0
    terminal_distinct = False
    actual_control_dt = None
    for _ in range(4):
        manager = randomization._newton_manager
        time_before = manager._sim_time if manager is not None else None
        tr = env.step(torch.zeros(env.N, 29, device=env.device))
        if manager is not None:
            actual_control_dt = manager._sim_time - time_before
            assert abs(actual_control_dt - 0.02) < 1e-8, actual_control_dt
        assert torch.isfinite(tr["s1"]).all()
        ref = env.reference.sample(tr["ref_id"], tr["ref_phase"] + 1, bodies=True)
        analytic = env.mdp.reward(unflatten_state(tr["s1"]), ref, tr["a"], tr["a_prev"])
        err = (analytic - tr["r"]).abs().max().item()
        reward_error = max(reward_error, err)
        assert err < 1e-5, err
        done = tr["done"]
        terminal_count += int(done.sum())
        if done.any():
            terminal_distinct |= bool((tr["s1"][done] - flatten_state(env.state())[done]).abs().max() > 1e-5)
    assert terminal_count > 0 and terminal_distinct, "Terminal transitions replaced by reset or no reset exercised"
    self_collision_values = []
    solver_exclusions = None
    solver_friction_error = None
    if randomization._newton_manager is not None:
        import warp as wp

        manager = randomization._newton_manager
        solver = manager._solver
        from isaaclab.sim import get_current_stage
        for prim in get_current_stage().Traverse():
            attr = prim.GetAttribute("newton:selfCollisionEnabled")
            if attr and "/Robot/" in str(prim.GetPath()):
                self_collision_values.append(bool(attr.Get()))
        assert self_collision_values and all(v == env.env.cfg.self_collisions for v in self_collision_values)
        solver_exclusions = int(solver.mj_model.nexclude)
        # Deliberate endpoints detect a fixed ground coefficient masking the
        # low end. This runs only in the disposable check environment.
        env.env.xi[0, 0] = 0.3
        if env.N > 1:
            env.env.xi[1, 0] = 1.2
        randomization._apply_material(ids)
        env.env.sim.step(render=False)
        friction = wp.to_torch(solver.mjw_model.geom_friction)[..., 0]
        body_ids = torch.as_tensor(solver.mj_model.geom_bodyid.copy(), device=env.device)
        ground = body_ids == 0
        robot_geoms = body_ids > 0
        assert ground.any() and robot_geoms.any()
        solver_friction_error = (friction[:, robot_geoms] - env.env.xi[:, :1]).abs().max().item()
        assert solver_friction_error < 1e-6, solver_friction_error
        assert torch.all(friction[:, ground] <= 0.3 + 1e-6)
    return {
        "environment": "SuperMimicRLEnv",
        "managerbased": True,
        "backend": getattr(env.env.cfg, "backend", "ovphysx"),
        "control_dt": env.env.step_dt,
        "actual_control_dt": actual_control_dt,
        "newton_self_collision_values": self_collision_values,
        "solver_excluded_body_pairs": solver_exclusions,
        "solver_friction_max_error": solver_friction_error,
        "torso_com_max_error": com_error,
        "friction_max_error": friction_error,
        "num_envs": env.N,
        "action_dim": 29,
        "observation_dim": 93,
        "fk_position_max_m": pos_error,
        "fk_rotation_max": rot_error,
        "partial_reset_max_error": partial_error,
        "real_imagined_reward_max_error": reward_error,
        "terminal_transitions": terminal_count,
        "terminal_state_preserved": terminal_distinct,
    }
