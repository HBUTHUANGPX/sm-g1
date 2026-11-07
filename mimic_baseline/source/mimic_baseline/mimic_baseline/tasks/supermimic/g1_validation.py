"""Real GPU assertions for the original manager environment, not a simulator double."""
import torch
from .mdp_legacy import flatten_state, unflatten_state
from .rotations import quat_to_mat

@torch.no_grad()
def check_environment(env):
    from isaaclab.envs import ManagerBasedRLEnv
    assert isinstance(env.env,ManagerBasedRLEnv)
    assert env.action_manager.total_action_dim == 29
    ids=torch.arange(env.N,device=env.device)
    frames=env.motion.clip_start[ids % len(env.motion.names)]
    env.reset(ids,frames=frames)
    assert env.policy_observation().shape == (env.N,93)
    kin=env.mdp.kin;robot=env.scene['robot']
    bp,bR,_,_=kin.forward(*env.state())
    body_ids=[robot.body_names.index(n) for n in kin.tracked_body_names]
    poses=robot.data.body_link_pose_w.torch[:,body_ids]
    sim_R=quat_to_mat(poses[...,[6,3,4,5]])
    pos_error=(bp-(poses[...,:3]-env.scene.env_origins[:,None])).norm(dim=-1).max().item()
    rot_error=(bR-sim_R).abs().max().item()
    assert pos_error < 1e-3 and rot_error < 1e-3,(pos_error,rot_error)
    before=flatten_state(env.state()).clone()
    env.reset(ids[:1],frames=frames[:1])
    partial_error=(flatten_state(env.state())[1:]-before[1:]).abs().max().item() if env.N>1 else 0.
    assert partial_error < 1e-5,partial_error
    terminal_count=0;reward_error=0.;terminal_distinct=False
    for _ in range(4):
        tr=env.step(torch.zeros(env.N,29,device=env.device))
        assert torch.isfinite(tr['s1']).all()
        ref=env.reference.sample(tr['ref_id'],tr['ref_phase']+1,bodies=True)
        analytic=env.mdp.reward(unflatten_state(tr['s1']),ref,tr['a'],tr['a_prev'])
        err=(analytic-tr['r']).abs().max().item();reward_error=max(reward_error,err)
        assert err < 1e-5,err
        done=tr['done'];terminal_count+=int(done.sum())
        if done.any():
            terminal_distinct |= bool((tr['s1'][done]-flatten_state(env.state())[done]).abs().max()>1e-5)
    assert terminal_count>0 and terminal_distinct,'Terminal transitions replaced by reset or no reset exercised'
    return {'environment':'SuperMimicRLEnv','managerbased':True,'num_envs':env.N,
        'action_dim':29,'observation_dim':93,'fk_position_max_m':pos_error,'fk_rotation_max':rot_error,
        'partial_reset_max_error':partial_error,'real_imagined_reward_max_error':reward_error,
        'terminal_transitions':terminal_count,'terminal_state_preserved':terminal_distinct}
