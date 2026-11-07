import numpy as np
import torch
from mimic_baseline.tasks.supermimic.lafan1 import convert_clip


def test_lafan1_resampling_units_and_last_wrist(tmp_path):
    raw = np.zeros((4,36), dtype=np.float32)
    raw[:,0] = np.arange(4) / 30
    raw[:,6] = 1 # source XYZW
    raw[:,35] = np.arange(4) / 30 * 2
    source=tmp_path/'walk.csv'; target=tmp_path/'walk.npz'
    np.savetxt(source,raw,delimiter=',')
    convert_clip(source,target)
    with np.load(target,allow_pickle=False) as data:
        assert data['joint_pos'].shape == (6,29)
        np.testing.assert_allclose(data['joint_pos'][:,-1],np.arange(6)*.04,atol=1e-6)
        np.testing.assert_allclose(data['joint_vel'][:,-1],2.,atol=1e-5)
        np.testing.assert_allclose(data['body_lin_vel_w'][:,0,0],1.,atol=1e-5)
        np.testing.assert_allclose(data['body_quat_w'][:,0],np.tile([1,0,0,0],(6,1)))
        assert str(data['robot_name']) == 'unitree_g1'
        assert str(data['body_names'][0]) == 'pelvis'


def test_original_motion_library_loads_g1_without_neck_padding(tmp_path):
    from test_kinematics import make_urdf
    from mimic_baseline.tasks.supermimic.g1 import G1Kinematics
    from mimic_baseline.tasks.supermimic.motion import MotionLib
    from mimic_baseline.tasks.supermimic.reference import StaticReference
    raw=np.zeros((4,36)); raw[:,6]=1.; raw[:,35]=.4
    source=tmp_path/'walk.csv'; target=tmp_path/'walk.npz'
    np.savetxt(source,raw,delimiter=','); convert_clip(source,target)
    kin=G1Kinematics(make_urdf(tmp_path),'cpu')
    motion=MotionLib([target],kin,'cpu')
    assert motion.action_dim == 29
    assert torch.allclose(motion.q[:,-1],torch.full_like(motion.q[:,-1],.4))
    batch=StaticReference(motion).sample(torch.tensor([0]),torch.tensor([0]),goal_frames=2)
    assert batch.goal().shape == (1,146)
    assert motion.original_clip_len.tolist() == [6]


def test_changed_source_rebuilds_cache_and_invalid_source_preserves_old_file(tmp_path):
    import pytest
    from mimic_baseline.tasks.supermimic.lafan1 import ensure_clip
    source=tmp_path/'walk.csv';target=tmp_path/'walk.npz'
    raw=np.zeros((4,36));raw[:,6]=1.
    np.savetxt(source,raw,delimiter=',')
    first=ensure_clip(source,target)
    raw[:,35]=.7;np.savetxt(source,raw,delimiter=',')
    second=ensure_clip(source,target)
    assert first != second
    with np.load(target) as data:np.testing.assert_allclose(data['joint_pos'][:,-1],.7)
    saved=target.read_bytes();source.write_text('invalid')
    with pytest.raises(ValueError):ensure_clip(source,target)
    assert target.read_bytes() == saved
