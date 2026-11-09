"""Ancestor-pruned FK preserves tracked values and every upstream gradient."""
from copy import copy
import os

import pytest
import torch

from test_training_kinematics import fixture_urdf, inputs_for, cast
from mimic_baseline.tasks.supermimic.core.kinematics import RXP2Kinematics
from mimic_baseline.tasks.supermimic.robots.g1 import kinematics as g1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch', [(), (9,), (2, 3)])
def test_lean_matches_reference_fixed_branches_and_gradients(tmp_path, dtype, batch):
    assert hasattr(g1, 'G1LeanKinematics'), 'ancestor-pruned batched FK is missing'
    kin = cast(g1.G1LeanKinematics(fixture_urdf(tmp_path), 'cpu'), dtype)
    inputs = inputs_for(batch, dtype)
    actual, reference = kin.forward(*inputs), RXP2Kinematics.forward(kin, *inputs)
    tolerance = 2e-4 if dtype == torch.float32 else 1e-11
    weights = [torch.linspace(-1, 1, x.numel(), dtype=dtype).reshape(x.shape) for x in actual]
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    gradients = [torch.autograd.grad(sum((x*w).sum() for x, w in zip(out, weights)), inputs)
                 for out in (actual, reference)]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


def test_lean_prunes_unused_joint_computation(tmp_path):
    assert hasattr(g1, 'G1LeanKinematics'), 'ancestor-pruned batched FK is missing'
    kin = g1.G1LeanKinematics(fixture_urdf(tmp_path), 'cpu')
    inputs = inputs_for((3,), torch.float32)
    expected = kin.forward(*inputs)
    # Untracked leaf branches deliberately contain NaNs; reference tracked
    # outputs are independent of these joints and must remain finite.
    required = set(kin.tracked_idx)
    for i in kin.tracked_idx:
        while kin.parent[i] >= 0:
            i = kin.parent[i]
            required.add(i)
    unused = [j for i, j in enumerate(kin.joint_idx) if j >= 0 and i not in required]
    assert unused
    inputs[4] = inputs[4].detach().clone()
    inputs[5] = inputs[5].detach().clone()
    inputs[4][:, unused] = float('nan')
    inputs[5][:, unused] = float('nan')
    inputs[4].requires_grad_()
    inputs[5].requires_grad_()
    actual = kin.forward(*inputs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    gradients = torch.autograd.grad(sum(x.sum() for x in actual), inputs)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert not gradients[4][:, unused].count_nonzero()
    assert not gradients[5][:, unused].count_nonzero()


def test_lean_copy_double_gradcheck(tmp_path):
    assert hasattr(g1, 'G1LeanKinematics'), 'ancestor-pruned batched FK is missing'
    original = g1.G1LeanKinematics(fixture_urdf(tmp_path), 'cpu')
    original.forward(*inputs_for((2,), torch.float32))
    kin = cast(copy(original), torch.float64)
    assert original.off_R.dtype == torch.float32
    assert kin.forward.__self__ is kin
    assert torch.autograd.gradcheck(kin.forward, inputs_for((2,)), fast_mode=True)


def test_lean_native_values_and_gradients():
    path = os.environ.get('G1_TEST_URDF')
    if not path:
        pytest.skip('Set G1_TEST_URDF to validate native asset')
    device = os.environ.get('G1_TEST_DEVICE', 'cpu')
    kin = g1.G1LeanKinematics(path, device)
    inputs = inputs_for((4096,), torch.float32, device)
    actual, reference = kin.forward(*inputs), RXP2Kinematics.forward(kin, *inputs)
    weights = [torch.linspace(-1, 1, x.numel(), device=device).reshape(x.shape) for x in actual]
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    grads = [torch.autograd.grad(sum((x*w).sum() for x, w in zip(out, weights)), inputs)
             for out in (actual, reference)]
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-4)


def test_lean_native_retained_storage_near_reference():
    """Redundant level-parent copies must not inflate saved FK activations."""
    import weakref

    path = os.environ.get('G1_TEST_URDF')
    if not path:
        pytest.skip('Set G1_TEST_URDF to validate native saved storage')
    kin = g1.G1LeanKinematics(path, 'cpu')
    inputs = inputs_for((1024,), torch.float32)

    class Saved:
        def __init__(self, tensor):
            self.tensor = tensor

    def live_storage(method):
        saved = []

        def pack(tensor):
            holder = Saved(tensor)
            saved.append(weakref.ref(holder))
            return holder

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda holder: holder.tensor):
            outputs = method(*inputs)
        storage = {ref().tensor.untyped_storage().data_ptr(): ref().tensor.untyped_storage().nbytes()
                   for ref in saved if ref() is not None}
        total = sum(storage.values())
        assert len(outputs) == 4
        return total

    reference = live_storage(lambda *xs: RXP2Kinematics.forward(kin, *xs))
    assert live_storage(kin.forward) <= reference * 1.15
