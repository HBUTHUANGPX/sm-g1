"""Native CUDA differential check; no simulator rollout or learning involved."""

import argparse
import json
from types import SimpleNamespace


def validate(urdf, device="cuda:0", *, training=False):
    import torch
    from isaaclab.utils.types import ArticulationActions
    from ..robots.g1.asset import make_g1_robot_cfg
    from ..envs.evaluation_actuators import group_evaluation_actuators
    from ..envs.grouped_actuators import group_g1_actuators

    comparisons = 0
    for max_delay in (0, 4):
        robot, _ = make_g1_robot_cfg(
            "unused.usda", urdf, actuator_type="delayed_pd", max_delay=max_delay
        )
        old_cfg = robot.actuators.copy()
        draw_order = list(old_cfg)
        order = list(reversed(draw_order))
        cfg = SimpleNamespace(
            eval_mode=not training,
            actuator_type="delayed_pd",
            scene=SimpleNamespace(robot=robot),
        )
        if training:
            group_g1_actuators(cfg)
        else:
            group_evaluation_actuators(cfg)
        c = next(iter(robot.actuators.values()))
        n, j = 40, len(order)
        old = {
            name: a.class_type(
                a,
                joint_names=[name],
                joint_ids=torch.tensor([order.index(name)], device=device),
                num_envs=n,
                device=device,
            )
            for name, a in old_cfg.items()
        }
        grouped = c.class_type(
            c, joint_names=order, joint_ids=slice(None), num_envs=n, device=device
        )
        for step in range(40):
            if step in (0, 9, 15, 23, 31):
                ids = {
                    0: None,
                    9: torch.tensor([1, 5, 13], device=device),
                    15: torch.empty(0, dtype=torch.long, device=device),
                    23: torch.tensor([13, 2, 7], device=device),
                    31: slice(None),
                }[step]
                torch.manual_seed(93 + step)
                for a in old.values():
                    a.reset(ids)
                rng = (
                    torch.cuda.get_rng_state(device)
                    if device.startswith("cuda")
                    else torch.get_rng_state()
                )
                torch.manual_seed(93 + step)
                grouped.reset(ids)
                now = (
                    torch.cuda.get_rng_state(device)
                    if device.startswith("cuda")
                    else torch.get_rng_state()
                )
                assert torch.equal(rng, now), "RNG consumption changed"
                expected = torch.stack(
                    [old[name].positions_delay_buffer.time_lags for name in order], 1
                )
                torch.testing.assert_close(
                    grouped.positions_delay_buffer.time_lags, expected, rtol=0, atol=0
                )
            if training and step in (5, 9, 23):
                # Exercise native tensor writes in resolved simulator joint order.
                # Current training randomizes material/COM/calibration, not gains;
                # this additionally guards future joint-indexed gain/limit writes.
                rows = torch.tensor([1, 5, 13], device=device)
                for field in ("stiffness", "damping", "actuator_effort_limit", "actuator_velocity_limit"):
                    values = torch.rand(len(rows), j, device=device) * 10 + 0.1
                    getattr(grouped, field)[rows] = values
                    for k, name in enumerate(order):
                        getattr(old[name], field)[rows] = values[:, k:k+1]
            for field in ("stiffness", "damping", "actuator_effort_limit", "actuator_velocity_limit"):
                torch.testing.assert_close(
                    getattr(grouped, field),
                    torch.cat([getattr(old[name], field) for name in order], 1),
                    rtol=0, atol=0,
                )
            values = [torch.randn(n, j, device=device) * 3 for _ in range(5)]
            pos, vel, effort, q, qd = values
            sync = step in (0, 9, 15, 23, 31)
            grouped.reset_sync_only = sync
            out = grouped.compute(
                ArticulationActions(
                    joint_positions=pos.clone(),
                    joint_velocities=vel.clone(),
                    joint_efforts=effort.clone(),
                ),
                q,
                qd,
            )
            expected = []
            computed = []
            for k, name in enumerate(order):
                a = old[name]
                a.reset_sync_only = sync
                r = a.compute(
                    ArticulationActions(
                        joint_positions=pos[:, k : k + 1].clone(),
                        joint_velocities=vel[:, k : k + 1].clone(),
                        joint_efforts=effort[:, k : k + 1].clone(),
                    ),
                    q[:, k : k + 1],
                    qd[:, k : k + 1],
                )
                expected.append(r.joint_efforts)
                computed.append(a.computed_effort)
            torch.testing.assert_close(
                out.joint_efforts, torch.cat(expected, 1), rtol=0, atol=0
            )
            torch.testing.assert_close(
                grouped.computed_effort, torch.cat(computed, 1), rtol=0, atol=0
            )
            comparisons += n * j
    return {
        "passed": True,
        "training": training,
        "gain_and_limit_writes": training,
        "empty_and_full_slice_resets": True,
        "device": device,
        "exact_joint_transition_comparisons": comparisons,
        "delay_bounds": [0, 4],
        "partial_resets": True,
        "reset_sync_only": True,
        "rng_state_identical": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--training", action="store_true")
    a = p.parse_args()
    print(json.dumps(validate(a.urdf, a.device, training=a.training)))


if __name__ == "__main__":
    main()
