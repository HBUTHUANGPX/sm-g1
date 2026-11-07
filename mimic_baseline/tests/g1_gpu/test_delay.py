import pytest
import torch
from isaaclab.actuators import DelayedPDActuator
from isaaclab.utils.types import ArticulationActions
from mimic_baseline.tasks.supermimic.actuators import G1DelayedPDActuator, G1DelayedPDActuatorCfg


@pytest.mark.parametrize("lag", [0, 2, 4])
def test_delayed_pd_matches_native_and_reset_writes_do_not_advance(lag):
    cfg = G1DelayedPDActuatorCfg(
        joint_names_expr=["joint"],
        stiffness=30.0,
        damping=2.0,
        effort_limit=50.0,
        effort_limit_sim=50.0,
        min_delay=lag,
        max_delay=lag,
    )
    native = DelayedPDActuator(cfg.copy(), ["joint"], slice(None), 4, "cuda:0")
    new = G1DelayedPDActuator(cfg.copy(), ["joint"], slice(None), 4, "cuda:0")

    def compute(a, target):
        zeros = torch.zeros_like(target)
        return a.compute(
            ArticulationActions(
                joint_positions=target.clone(), joint_velocities=zeros.clone(), joint_efforts=zeros.clone()
            ),
            zeros,
            zeros,
        ).joint_efforts

    for step in range(15):
        if step in (0, 8):
            ids = None if step == 0 else torch.tensor([0, 2], device="cuda:0")
            native.reset(ids)
            new.reset(ids)
            head = new.positions_delay_buffer._head
            lengths = new.positions_delay_buffer._lengths.clone()
            new.reset_sync_only = True
            compute(new, torch.full((4, 1), 9.0, device="cuda:0"))
            new.reset_sync_only = False
            assert new.positions_delay_buffer._head == head
            torch.testing.assert_close(new.positions_delay_buffer._lengths, lengths)
        target = torch.full((4, 1), step * 0.05, device="cuda:0")
        torch.testing.assert_close(compute(new, target), compute(native, target), rtol=0, atol=0)
    for _ in range(lag + 1):
        saturated = compute(new, torch.full((4, 1), 100.0, device="cuda:0"))
    assert (saturated == 50).all()
    assert not hasattr(new, "_torque_speed_limiter")


def test_g1_factory_selects_delayed_pd_without_tn(tmp_path):
    from mimic_baseline.tasks.supermimic.g1 import G1_JOINT_NAMES
    from mimic_baseline.tasks.supermimic.g1_robot import make_g1_robot_cfg

    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        "<robot>"
        + "".join(f'<joint name="{n}"><limit effort="50" velocity="10"/></joint>' for n in G1_JOINT_NAMES)
        + "</robot>"
    )
    cfg, scales = make_g1_robot_cfg("robot.usda", urdf, actuator_type="delayed_pd", min_delay=0, max_delay=4)
    assert len(scales) == len(cfg.actuators) == 29
    for actuator in cfg.actuators.values():
        assert actuator.class_type is G1DelayedPDActuator
        assert actuator.min_delay == 0 and actuator.max_delay == 4
        assert actuator.effort_limit == 50
    with pytest.raises(ValueError, match="delay"):
        make_g1_robot_cfg("robot.usda", urdf, max_delay=4)
