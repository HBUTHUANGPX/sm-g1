import torch
from mimic_baseline.tasks.supermimic.kinematics import RXP2Kinematics
from mimic_baseline.tasks.supermimic.g1 import G1Kinematics, G1_JOINT_NAMES, G1_TRACKED_BODY_NAMES


def make_urdf(tmp_path):
    assert issubclass(G1Kinematics, RXP2Kinematics)
    assert len(G1_JOINT_NAMES) == len(set(G1_JOINT_NAMES)) == 29
    assert G1_JOINT_NAMES[-1] == "right_wrist_yaw_joint"
    bodies = list(G1_TRACKED_BODY_NAMES[1:])
    bodies += [f"extra_{i}" for i in range(29 - len(bodies))]
    xml = ['<robot name="g1"><link name="pelvis"/>']
    parent = "pelvis"
    for joint, body in zip(G1_JOINT_NAMES, bodies):
        xml.append(
            f'<link name="{body}"/><joint name="{joint}" type="revolute"><parent link="{parent}"/><child link="{body}"/><origin xyz="0.1 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>'
        )
        parent = body
    path = tmp_path / "g1.urdf"
    path.write_text("".join(xml) + "</robot>")
    return path


def test_g1_uses_original_fk_and_all_policy_joints(tmp_path):
    kin = G1Kinematics(make_urdf(tmp_path), "cpu")
    assert kin.action_dim == 29
    assert kin.root_body_name == "pelvis"
    q = torch.zeros(2, 29, requires_grad=True)
    p = torch.zeros(2, 3)
    R = torch.eye(3).expand(2, 3, 3)
    out = kin.forward_poses(p, R, q)
    assert torch.isfinite(out[0]).all()
    out[1][:, -1, 1, 0].sum().backward()
    assert torch.all(q.grad[:, -1].abs() > 0.9)


def test_fast_forward_values_and_gradients(tmp_path):
    """Branching, fixed links, nontrivial origins and every input gradient."""
    import xml.etree.ElementTree as ET

    path = make_urdf(tmp_path)
    tree = ET.parse(path)
    root = tree.getroot()
    joints = root.findall('joint')
    for i, joint in enumerate(joints):
        # Mix chains and siblings; retain all canonical/tracked names.
        if i % 3 == 0:
            joint.find('parent').set('link', 'pelvis')
        joint.find('origin').set('rpy', '0.12 -0.21 0.33')
        joint.find('origin').set('xyz', '0.13 -0.08 0.04')
        joint.find('axis').set('xyz', ('1 0 0', '0 1 0', '0 0 1')[i % 3])
    fixed = ET.SubElement(root, 'joint', name='fixed_test', type='fixed')
    ET.SubElement(root, 'link', name='fixed_test_link')
    ET.SubElement(fixed, 'parent', link='pelvis')
    ET.SubElement(fixed, 'child', link='fixed_test_link')
    ET.SubElement(fixed, 'origin', xyz='0.2 0.1 -0.1', rpy='0.2 0.3 -0.1')
    joints[0].find('parent').set('link', 'fixed_test_link')
    tree.write(path)
    kin = G1Kinematics(path, 'cpu')
    assert not hasattr(kin, '_fast_levels')
    for name in ('off_p', 'off_R', 'axes'):
        setattr(kin, name, getattr(kin, name).double())
    generator = torch.Generator().manual_seed(91)
    for batch in ((), (4,), (2, 3)):
        inputs = [torch.randn(*batch, *shape, generator=generator, dtype=torch.float64,
                              requires_grad=True)
                  for shape in ((3,), (3, 3), (3,), (3,), (29,), (29,))]
        reference = kin.forward(*inputs)
        actual = kin.forward_fast(*inputs)
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        weights = [torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in actual]
        ref_grad = torch.autograd.grad(sum((x * z).sum() for x, z in zip(reference, weights)), inputs)
        fast_grad = torch.autograd.grad(sum((x * z).sum() for x, z in zip(actual, weights)), inputs)
        for a, b in zip(fast_grad, ref_grad):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    assert torch.autograd.gradcheck(kin.forward_fast, inputs, fast_mode=True)


def test_fast_forward_rejects_invalid_joint_dimensions(tmp_path):
    import pytest

    kin = G1Kinematics(make_urdf(tmp_path), 'cpu')
    p = torch.zeros(3)
    with pytest.raises(ValueError, match='Expected q and qd'):
        kin.forward_fast(p, torch.eye(3), p, p, torch.zeros(28), torch.zeros(28))
    with pytest.raises(ValueError, match='Expected q and qd'):
        kin.forward_fast(p, torch.eye(3), p, p, torch.zeros(29), torch.zeros(1, 29))


def test_fast_forward_rejects_non_breadth_first_topology(tmp_path):
    import pytest

    kin = G1Kinematics(make_urdf(tmp_path), 'cpu')
    # Keep a connected tree but visit body 3 before body 2 by depth.
    parents = list(kin.parent)
    parents[3] = 0
    kin.parent = tuple(parents)
    with pytest.raises(ValueError, match='breadth-first'):
        kin._initialize_fast_topology()


def test_batched_copy_uses_copied_double_transforms(tmp_path):
    from copy import copy
    from mimic_baseline.tasks.supermimic.robots.g1.kinematics import G1BatchedKinematics

    original = G1BatchedKinematics(make_urdf(tmp_path), 'cpu')
    copied = copy(original)
    for name in ('off_p', 'off_R', 'axes', 'q_lo', 'q_hi'):
        setattr(copied, name, getattr(original, name).double())
    assert copied.forward.__self__ is copied
    assert original.forward.__self__ is original
    assert original.off_R.dtype == torch.float32
    assert copied.off_R.dtype == torch.float64
    p = torch.zeros(2, 3, dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    q = torch.zeros(2, 29, dtype=torch.float64)
    actual = copied.forward(p, R, p, p, q, q)
    reference = RXP2Kinematics.forward(copied, p, R, p, p, q, q)
    for a, b in zip(actual, reference):
        assert a.dtype == torch.float64
        torch.testing.assert_close(a, b)


def test_native_g1_values_gradients_and_motionlib(tmp_path):
    """Optional native-asset validation on CPU or the explicitly selected GPU."""
    import os
    import numpy as np
    import pytest
    from mimic_baseline.tasks.supermimic.robots.g1.kinematics import G1BatchedKinematics
    from mimic_baseline.tasks.supermimic.data.lafan1 import convert_clip
    from mimic_baseline.tasks.supermimic.data.motion import MotionLib
    from mimic_baseline.tasks.supermimic.core.rotations import quat_to_mat

    path = os.environ.get('G1_TEST_URDF')
    if not path:
        pytest.skip('Set G1_TEST_URDF to validate native G1 asset')
    device = os.environ.get('G1_TEST_DEVICE', 'cpu')
    kin = G1BatchedKinematics(path, device)
    torch.manual_seed(19)
    for batch in (40, 4096):
        inputs = [torch.randn(batch, *shape, device=device, requires_grad=True)
                  for shape in ((3,), (3, 3), (3,), (3,), (29,), (29,))]
        inputs[1] = quat_to_mat(torch.nn.functional.normalize(
            torch.randn(batch, 4, device=device), dim=-1)).detach().requires_grad_()
        reference = RXP2Kinematics.forward(kin, *inputs)
        actual = kin.forward(*inputs)
        for a, b in zip(actual, reference):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        weights = [torch.randn_like(x) for x in actual]
        ref_grad = torch.autograd.grad(sum((x * z).sum() for x, z in zip(reference, weights)), inputs)
        fast_grad = torch.autograd.grad(sum((x * z).sum() for x, z in zip(actual, weights)), inputs)
        for a, b in zip(fast_grad, ref_grad):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
        print('native G1', device, 'batch', batch,
              'output_max_abs', max((a-b).abs().max().item() for a, b in zip(actual, reference)),
              'gradient_max_abs', max((a-b).abs().max().item() for a, b in zip(fast_grad, ref_grad)))
    raw = np.zeros((5, 36))
    raw[:, 2] = 0.8
    raw[:, 6] = 1
    raw[:, 7:] = np.linspace(0, 0.1, 5)[:, None]
    source = tmp_path / 'native.csv'
    target = source.with_suffix('.npz')
    np.savetxt(source, raw, delimiter=',')
    convert_clip(source, target)
    motion = MotionLib([target], kin, device)
    assert motion.kin.forward.__self__ is kin
    assert motion.kin.off_R.dtype == torch.float32
    assert motion.root_p.dtype == torch.float32
    assert torch.isfinite(motion.root_p).all()
