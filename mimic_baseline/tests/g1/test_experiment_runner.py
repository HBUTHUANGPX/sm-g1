import importlib.util
import json
import hashlib
from pathlib import Path
import subprocess
import torch
import pytest

spec = importlib.util.spec_from_file_location('runner', Path(__file__).parents[2] / 'scripts/supermimic/run_g1_experiment.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('failure', [None, 'training', 'checkpoint', 'evaluating', 'clip_set', 'source_drift'])
def test_experiment_gate_and_complete_clip_set(tmp_path, monkeypatch, failure):
    data = tmp_path / 'data/lafan1_g1'
    data.mkdir(parents=True)
    for i in range(40):
        (data / f'clip{i}.csv').touch()
    source = tmp_path / 'source'
    source.mkdir()
    tracked = source / 'train.py'; tracked.write_text('original')
    manifest = {'source_revision':'0'*40, 'files':{'train.py':hashlib.sha256(tracked.read_bytes()).hexdigest()}}
    (source / 'release-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, 'PROJECT_ROOT', source)
    calls = []
    def child(command, **kwargs):
        mode = command[command.index('--mode') + 1]
        calls.append(mode)
        if (mode == 'train' and failure == 'training') or (mode == 'evaluate' and failure == 'evaluating'):
            raise subprocess.CalledProcessError(1, command)
        out = Path(command[command.index('--log-dir') + 1]); out.mkdir()
        if mode == 'train':
            cfg = dict(recovery_enabled=True, self_collisions=True, actuator_type='delayed_pd', max_delay=4, num_envs=64)
            ck = dict(iteration=2 if failure == 'checkpoint' else 3, format='supermimic-g1-v1', config=cfg, provenance={'backend':'newton'})
            ck.update({k:{'weight':torch.ones(1)} for k in ('wm','policy','critic')})
            torch.save(ck, out / 'model_3.pt')
            if failure == 'source_drift': tracked.write_text('modified')
        else:
            rows = [dict(name=f'clip{i}', completed=False) for i in range(39 if failure == 'clip_set' else 40)]
            (out / 'perclip.json').write_text(json.dumps({'clips':rows}))
    monkeypatch.setattr(runner.subprocess, 'run', child)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            runner.run_experiment(data.parent, tmp_path/'run', 64, 3, '0'*40)
    else:
        result = runner.run_experiment(data.parent, tmp_path/'run', 64, 3, '0'*40)
        assert result['phase']=='complete' and result['evaluated_clips']==40 and result['completed_clips']==0
    status = json.loads((tmp_path/'run/status.json').read_text())
    if failure in ('training', 'checkpoint'):
        assert calls == ['train'] and status['phase']=='training_failed'
    elif failure:
        assert status['phase']=='evaluating_failed'
    tracked.write_text('original')
    with pytest.raises(FileExistsError):
        runner.run_experiment(data.parent, tmp_path/'run', 64, 3, '0'*40)


def test_source_revision_mismatch_is_rejected(tmp_path, monkeypatch):
    (tmp_path/'release-manifest.json').write_text(json.dumps({'source_revision':'1'*40,'files':{'file':'digest'}}))
    monkeypatch.setattr(runner,'PROJECT_ROOT',tmp_path)
    with pytest.raises(ValueError,match='revision mismatch'):
        runner.verify_source('0'*40)
