import pytest
import torch
from mimic_baseline.robots.actuator_delay_buffer import ActuatorDelayBuffer
from mimic_baseline.tasks.supermimic.evaluation.joint_delay import (
    JointDelayBuffer,
    sample_joint_delays,
)


@pytest.mark.parametrize("max_delay", [0, 4])
def test_batched_delay_matches_independent_joints_with_partial_resets(max_delay):
    n, j = 7, 29
    grouped = JointDelayBuffer(max_delay, n, j, "cpu")
    separate = [ActuatorDelayBuffer(max_delay, n, "cpu") for _ in range(j)]
    torch.manual_seed(17)
    lag = torch.randint(max_delay + 1, (n, j))
    grouped.set_time_lag(lag)
    for k, b in enumerate(separate):
        b.set_time_lag(lag[:, k])
    for step in range(21):
        if step in (3, 9):
            ids = torch.tensor([1, 4])
            grouped.reset(ids)
            for b in separate:
                b.reset(ids)
        value = torch.randn(n, j)
        actual = grouped.compute(value)
        expected = torch.cat(
            [b.compute(value[:, k : k + 1]) for k, b in enumerate(separate)], dim=1
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_delay_sampling_keeps_original_rng_order_in_arbitrary_joint_order():
    original = ["hip", "knee", "ankle"]
    resolved = ["ankle", "hip", "knee"]
    torch.manual_seed(23)
    expected = {name: torch.randint(0, 5, (8,), dtype=torch.int) for name in original}
    next_expected = torch.rand(3)
    torch.manual_seed(23)
    actual = sample_joint_delays(original, resolved, 8, 0, 4, "cpu")
    torch.testing.assert_close(
        actual, torch.stack([expected[name] for name in resolved], 1), rtol=0, atol=0
    )
    torch.testing.assert_close(torch.rand(3), next_expected, rtol=0, atol=0)
