"""Training FK numerical/autograd contracts and activation-storage regression."""
from copy import copy
import os
import xml.etree.ElementTree as ET

import pytest
import torch

from test_kinematics import make_urdf
from mimic_baseline.tasks.supermimic.core.kinematics import RXP2Kinematics
from mimic_baseline.tasks.supermimic.robots.g1 import kinematics as g1


def fixture_urdf(tmp_path):
    path = make_urdf(tmp_path)
    tree = ET.parse(path)
    root = tree.getroot()
    joints = root.findall('joint')
    for i, joint in enumerate(joints):
        if i % 7 == 0:
            joint.find('parent').set('link', 'pelvis')
        joint.find('origin').set('rpy', '0.12 -0.21 0.33')
        joint.find('origin').set('xyz', '0.13 -0.08 0.04')
        joint.find('axis').set('xyz', ('1 0 0', '0 1 0', '0 0 1')[i % 3])
    for i in range(9):
        ET.SubElement(root, 'link', name=f'fixed_{i}')
        joint = ET.SubElement(root, 'joint', name=f'fixed_joint_{i}', type='fixed')
        ET.SubElement(joint, 'parent', link='pelvis' if i == 0 else f'fixed_{i-1}')
        ET.SubElement(joint, 'child', link=f'fixed_{i}')
        ET.SubElement(joint, 'origin', xyz='0.03 0.02 -0.01', rpy='0.2 -0.1 0.3')
    joints[0].find('parent').set('link', 'fixed_0')
    tree.write(path)
    return path


def inputs_for(batch, dtype=torch.float64, device='cpu'):
    generator = torch.Generator(device=device).manual_seed(431)
    return [torch.randn(*batch, *shape, dtype=dtype, device=device,
                        generator=generator, requires_grad=True)
            for shape in ((3,), (3, 3), (3,), (3,), (29,), (29,))]


def cast(kin, dtype):
    for name in ('off_p', 'off_R', 'axes', 'q_lo', 'q_hi'):
        setattr(kin, name, getattr(kin, name).to(dtype))
    return kin


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch', [(), (7,), (2, 4)])
def test_training_preserves_values_and_six_input_gradients(tmp_path, dtype, batch):
    assert hasattr(g1, 'G1TrainingKinematics'), 'memory-bounded training FK is missing'
    kin = cast(g1.G1TrainingKinematics(fixture_urdf(tmp_path), 'cpu', chunk_size=3), dtype)
    inputs = inputs_for(batch, dtype)
    reference = RXP2Kinematics.forward(kin, *inputs)
    actual = kin.forward(*inputs)
    tol = 2e-4 if dtype == torch.float32 else 1e-11
    weights = [torch.linspace(-1, 1, x.numel(), dtype=dtype).reshape(x.shape) for x in actual]
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    gradients = [torch.autograd.grad(sum((x*z).sum() for x, z in zip(out, weights)), inputs)
                 for out in (actual, reference)]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    with torch.no_grad():
        for a, b in zip(kin.forward(*inputs), reference):
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)


def test_training_retains_only_inputs_for_backward(tmp_path):
    assert hasattr(g1, 'G1TrainingKinematics'), 'memory-bounded training FK is missing'
    kin = g1.G1TrainingKinematics(fixture_urdf(tmp_path), 'cpu', chunk_size=13)
    inputs = inputs_for((128,), torch.float32)
    input_pointers = {x.untyped_storage().data_ptr() for x in inputs}
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: (saved.append(x) or x), lambda x: x):
        outputs = kin.forward(*inputs)
    # Removing checkpointing must fail this: intermediates cannot survive forward.
    assert all(x.numel() == 0 or x.untyped_storage().data_ptr() in input_pointers for x in saved)
    assert sum(x.numel()*x.element_size() for x in saved) == sum(x.numel()*x.element_size() for x in inputs)
    sum(x.sum() for x in outputs).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs)


def test_training_copy_double_gradcheck_and_rng(tmp_path):
    assert hasattr(g1, 'G1TrainingKinematics'), 'memory-bounded training FK is missing'
    original = g1.G1TrainingKinematics(fixture_urdf(tmp_path), 'cpu', chunk_size=2)
    kin = cast(copy(original), torch.float64)
    assert original.off_R.dtype == torch.float32
    inputs = inputs_for((3,))
    before = torch.random.get_rng_state().clone()
    assert torch.autograd.gradcheck(kin.forward, inputs, fast_mode=True)
    assert torch.equal(before, torch.random.get_rng_state())
    assert kin.forward.__self__ is kin


def test_training_native_g1(tmp_path):
    path = os.environ.get('G1_TEST_URDF')
    if not path:
        pytest.skip('Set G1_TEST_URDF to validate native asset')
    device = os.environ.get('G1_TEST_DEVICE', 'cpu')
    kin = g1.G1TrainingKinematics(path, device, chunk_size=1024)
    inputs = inputs_for((4096,), torch.float32, device)
    actual, reference = kin.forward(*inputs), RXP2Kinematics.forward(kin, *inputs)
    weights = [torch.linspace(-1, 1, x.numel(), device=device).reshape(x.shape) for x in actual]
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    grads = [torch.autograd.grad(sum((x*z).sum() for x, z in zip(out, weights)), inputs)
             for out in (actual, reference)]
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-4)
