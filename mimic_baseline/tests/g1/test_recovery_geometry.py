from types import SimpleNamespace
import torch
from mimic_baseline.tasks.supermimic.recovery import RecoveryPlacement


def test_g1_placement_requires_g1_bodies_and_uses_perturbed_pose(monkeypatch):
    from mimic_baseline.utils import collision_bounds
    bounds = {'pelvis': ((-.1,-.1,-.1),(.1,.1,.1)), 'torso_link': ((-.2,-.1,-.3),(.2,.1,.3))}
    monkeypatch.setattr(collision_bounds, 'read_collision_bounds', lambda path: bounds)
    kin = SimpleNamespace(device='cpu', body_names=['pelvis','torso_link'],
                          recovery_required_bodies=('pelvis','torso_link'))
    kin.forward_poses = lambda p,R,q: (p[:,None].expand(-1,2,-1), R[:,None].expand(-1,2,-1,-1))
    placement = RecoveryPlacement(kin, 'unused')
    p = torch.tensor([[1.,2.,3.],[2.,1.,4.]])
    R = torch.stack((torch.eye(3),torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])))
    q = torch.zeros(2,29)
    raised = placement.place(p,R,q)
    torch.testing.assert_close(placement.corners(raised,R,q)[...,2].amin((-1,-2)), torch.full((2,),.15),atol=1e-6,rtol=0)
    torch.testing.assert_close(raised[:,:2],p[:,:2])


def test_recovery_errors_follow_named_body_mapping():
    from mimic_baseline.tasks.supermimic.recovery import tracking_errors
    p = torch.zeros(1,3); R = torch.eye(3)[None]
    ref = SimpleNamespace(p=p,R=R,body_p=torch.zeros(1,5,3),body_R=R[:,None].repeat(1,5,1,1))
    ref.detach = lambda: ref
    body_p = ref.body_p.clone(); body_p[:,0,0] = 1.
    errors = tracking_errors((p,R),SimpleNamespace(now=ref),(body_p,ref.body_R),
                             body_indices=(2,(0,1),(3,4)))
    torch.testing.assert_close(errors,torch.tensor([[0.,0.,1.,0.]]))
