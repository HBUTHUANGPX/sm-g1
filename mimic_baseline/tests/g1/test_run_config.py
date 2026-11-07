import pytest
from mimic_baseline.tasks.supermimic.g1_run_config import restore_run_config


def checkpoint():
    return {'format':'supermimic-g1-v1','config':{'robot_name':'g1','num_envs':64,'horizon':8,'iterations':10,'log_dir':'old'},'provenance':{'small_model':True}}


def test_resume_preserves_saved_training_and_infers_architecture():
    cfg,small=restore_run_config(checkpoint(),{'iterations':12,'log_dir':'new'})
    assert cfg['horizon']==8 and cfg['iterations']==12 and small is True
    assert checkpoint()['config']['log_dir']=='old'


def test_resume_rejects_incompatible_model_and_implicit_training_changes():
    with pytest.raises(ValueError,match='small_model'):restore_run_config(checkpoint(),{},small_model=False)
    with pytest.raises(ValueError,match='horizon'):restore_run_config(checkpoint(),{'horizon':32})
    wrong=checkpoint();wrong['format']='supermimic-rx-p2-v1'
    with pytest.raises(ValueError,match='G1'):restore_run_config(wrong,{})
