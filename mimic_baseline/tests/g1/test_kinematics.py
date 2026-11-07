import torch
from mimic_baseline.tasks.supermimic.kinematics import RXP2Kinematics
from mimic_baseline.tasks.supermimic.g1 import G1Kinematics, G1_JOINT_NAMES, G1_TRACKED_BODY_NAMES


def test_g1_uses_original_fk_and_all_policy_joints(tmp_path):
    assert issubclass(G1Kinematics, RXP2Kinematics)
    assert len(G1_JOINT_NAMES) == len(set(G1_JOINT_NAMES)) == 29
    assert G1_JOINT_NAMES[-1] == 'right_wrist_yaw_joint'
    bodies = list(G1_TRACKED_BODY_NAMES[1:])
    bodies += [f'extra_{i}' for i in range(29 - len(bodies))]
    xml = ['<robot name="g1"><link name="pelvis"/>']
    parent = 'pelvis'
    for joint, body in zip(G1_JOINT_NAMES, bodies):
        xml.append(f'<link name="{body}"/><joint name="{joint}" type="revolute"><parent link="{parent}"/><child link="{body}"/><origin xyz="0.1 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>')
        parent = body
    path = tmp_path / 'g1.urdf'
    path.write_text(''.join(xml) + '</robot>')
    kin = G1Kinematics(path, 'cpu')
    assert kin.action_dim == 29
    assert kin.root_body_name == 'pelvis'
    q = torch.zeros(2, 29, requires_grad=True)
    p = torch.zeros(2, 3)
    R = torch.eye(3).expand(2, 3, 3)
    out = kin.forward_poses(p, R, q)
    assert torch.isfinite(out[0]).all()
    out[1][:, -1, 1, 0].sum().backward()
    assert torch.all(q.grad[:, -1].abs() > 0.9)
