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


@torch.no_grad()
def check_recovery_environment(env):
    """Exercise real reset geometry, delayed physics and the original hazard manager."""
    from .env_managed import SuperMimicTerminationManager
    from .actuators import G1DelayedPDActuator
    from .recovery import PROTECTION_STEPS
    assert env.N >= 32, "Recovery check needs at least 32 environments"
    managed = env.env
    assert isinstance(managed.termination_manager, SuperMimicTerminationManager)
    command = managed.command_manager.get_term("motion")
    actuators = list(env.scene["robot"].actuators.values())
    delayed = all(isinstance(a, G1DelayedPDActuator) for a in actuators)
    actuator = actuators[0]
    count = 0
    clearance_error = 0.
    upside_down = 0
    observed_lags = set()
    for _ in range(8):
        env.reset()
        role = managed.recovery_role
        p, rotation, v, w, q, qd = env.state()
        count += int(role.sum())
        if delayed:
            observed_lags.update(actuator.positions_delay_buffer.time_lags.tolist())
        if role.any():
            corners = command._recovery_placement.corners(p[role], rotation[role], q[role])
            error = (corners[..., 2].amin(dim=(-1, -2)) - .15).abs().max().item()
            clearance_error = max(clearance_error, error)
            assert error < 1e-4, error
            assert max(v[role].abs().max(), w[role].abs().max(), qd[role].abs().max()) < 1e-5
            assert (q[role] >= env.mdp.kin.q_lo - 1e-5).all()
            assert (q[role] <= env.mdp.kin.q_hi + 1e-5).all()
            upside_down += int((rotation[role, 2, 2] < 0).sum())
    fraction = count / (8 * env.N)
    assert .08 < fraction < .22 and upside_down > 0, (fraction, upside_down)
    if delayed:
        assert observed_lags == set(range(actuator.cfg.min_delay, actuator.cfg.max_delay + 1)), observed_lags
    before = flatten_state(env.state()).clone()
    role_before = managed.recovery_role.clone()
    if delayed:
        lengths = actuator.positions_delay_buffer._lengths.clone()
        head = actuator.positions_delay_buffer._head
    env.reset(torch.tensor([0], device=env.device))
    assert torch.equal(flatten_state(env.state())[1:], before[1:])
    assert torch.equal(managed.recovery_role[1:], role_before[1:])
    if delayed:
        assert head == actuator.positions_delay_buffer._head
        assert torch.equal(lengths[1:], actuator.positions_delay_buffer._lengths[1:])

    manager = managed.termination_manager
    # Controlled manager inputs isolate acceptance/protection from motion errors.
    # Restore every callback afterwards; the following rollout uses real errors.
    callbacks = [(cfg.func, cfg.params) for cfg in manager._term_cfgs]
    original_age = managed.episode_length_buf.clone()
    original_role = managed.recovery_role.clone()
    original_eval = managed.cfg.eval_mode
    try:
        for index, cfg in enumerate(manager._term_cfgs):
            cfg.func = lambda e, value=index in manager._tracking_indices: torch.full((e.num_envs,), value, device=e.device)
            cfg.params = {}
        managed.recovery_role.fill_(True)
        managed.episode_length_buf.fill_(PROTECTION_STEPS)
        assert not manager.compute().any(), "Protected pre-step age 199 terminated"
        for cfg in manager._term_cfgs:
            if cfg.time_out:
                func = cfg.func
                cfg.func = lambda e: torch.ones(e.num_envs, device=e.device, dtype=torch.bool)
                assert manager.compute().all(), "Protection masked a truncation"
                cfg.func = func
        managed.episode_length_buf.fill_(PROTECTION_STEPS + 1)
        accepted = torch.zeros(4, device=env.device)
        for _ in range(200):
            manager.compute()
            accepted += manager.accepted_tracking.sum(0)
        rates = accepted / (200 * env.N)
        torch.testing.assert_close(rates, rates.new_tensor([.02/3, .02/3, .04, .04]), atol=.006, rtol=0)
        managed.cfg.eval_mode = True
        managed.episode_length_buf.fill_(1)
        assert manager.compute().all(), "Evaluation must use raw deterministic failures"
    finally:
        for cfg, (func, params) in zip(manager._term_cfgs, callbacks):
            cfg.func, cfg.params = func, params
        managed.cfg.eval_mode = original_eval
        managed.episode_length_buf.copy_(original_age)
        managed.recovery_role.copy_(original_role)
    env.reset()
    transitions = 0
    manager_newton = managed.randomization._newton_manager
    start_time = manager_newton._sim_time if manager_newton is not None else None
    self_contacts = 0
    if manager_newton is not None:
        import warp as wp
        solver = manager_newton._solver
        geom_bodies = torch.as_tensor(solver.mj_model.geom_bodyid.copy(), device=env.device)
        armature = solver.mj_model.dof_armature[6:]
        if delayed:
            assert abs(armature - .03).max() < 1e-6
    for step in range(205):
        heads = [a.positions_delay_buffer._head for a in actuators] if delayed else []
        tr = env.step(torch.zeros(env.N, 29, device=env.device))
        assert torch.isfinite(tr["s1"]).all() and torch.isfinite(tr["r"]).all(), {"step":step, "bad_state_rows": (~torch.isfinite(tr["s1"]).all(-1)).nonzero().tolist(), "reward_finite":bool(torch.isfinite(tr["r"]).all()), "max_finite_state":tr["s1"][torch.isfinite(tr["s1"])].abs().max().item()}
        assert tr["recovery_role"].dtype == torch.bool
        transitions += int(tr["done"].sum())
        for a, previous in zip(actuators, heads):
            buf = a.positions_delay_buffer
            assert buf._head == (previous + managed.cfg.decimation) % buf._capacity
        if manager_newton is not None:
            n = int(wp.to_torch(solver.mjw_data.nacon)[0])
            geoms = wp.to_torch(solver.mjw_data.contact.geom)[:n].long()
            bodies = geom_bodies[geoms]
            self_contacts += int(((bodies > 0).all(-1) & (bodies[:, 0] != bodies[:, 1])).sum())
    if manager_newton is not None and managed.cfg.self_collisions:
        assert self_contacts > 0, "No robot-to-robot contact observed"
    elapsed = manager_newton._sim_time - start_time if manager_newton is not None else None
    if elapsed is not None:
        assert abs(elapsed - 205 * .02) < 1e-6, elapsed
    return {"backend": managed.cfg.backend, "num_envs": env.N, "recovery_reset_fraction": fraction,
            "upside_down_resets": upside_down, "clearance_max_error_m": clearance_error,
            "observed_delay_ticks": sorted(observed_lags), "manager_group_acceptance_rates": rates.tolist(),
            "protected_boundary_passed": True, "deterministic_eval_passed": True,
            "partial_reset_passed": True, "finite_rollout_steps": 205,
            "simulated_seconds": elapsed, "terminal_transitions": transitions, "self_contact_samples": self_contacts,
            "timeouts_bypass_protection": True}
