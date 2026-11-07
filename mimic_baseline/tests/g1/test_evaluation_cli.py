import importlib.util
import json
from pathlib import Path
import subprocess
import torch
import pytest

spec=importlib.util.spec_from_file_location('evaluation_cli',Path(__file__).parents[2]/'scripts/supermimic/evaluate_g1.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def fixture(tmp_path):
    data=tmp_path/'data';(data/'lafan1_g1').mkdir(parents=True)
    for i in range(40): (data/'lafan1_g1'/f'clip{i}.csv').touch()
    ck=tmp_path/'model_100.pt'
    torch.save({'format':'supermimic-g1-v1','iteration':100,'config':{'robot_name':'g1'},'policy':{'p':torch.ones(1)}},ck)
    return ck,data,tmp_path/'eval'


def test_dry_run_does_not_create_outputs_or_launch(tmp_path,monkeypatch):
    ck,data,out=fixture(tmp_path)
    monkeypatch.setattr(cli.subprocess,'run',lambda *a,**kw:pytest.fail('Unexpected simulator'))
    r=cli.evaluate(ck,data,out,dry_run=True)
    assert r['preflight_passed'] and len(r['expected_clips'])==40 and not out.exists()


@pytest.mark.parametrize('failure',[None,'child','incomplete','nonfinite'])
def test_evaluation_outputs_and_failure_gates(tmp_path,monkeypatch,failure):
    ck,data,out=fixture(tmp_path)
    def run(command,**kwargs):
        if failure=='child': raise subprocess.CalledProcessError(1,command)
        rows=[{'name':f'clip{i}','completed':i<3,'normalized_return':.5} for i in range(39 if failure=='incomplete' else 40)]
        if failure=='nonfinite': rows[0]['normalized_return']=float('nan')
        (out/'perclip.json').write_text(json.dumps({'clips':rows}))
    monkeypatch.setattr(cli.subprocess,'run',run)
    if failure:
        with pytest.raises((ValueError,subprocess.CalledProcessError)): cli.evaluate(ck,data,out)
        assert (out/'failure.json').exists() and not (out/'summary.json').exists()
    else:
        r=cli.evaluate(ck,data,out);assert r['completed']==3 and r['clips']==40
    with pytest.raises(FileExistsError): cli.evaluate(ck,data,out)
