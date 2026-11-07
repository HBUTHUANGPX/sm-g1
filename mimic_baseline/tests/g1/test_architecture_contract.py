"""Frozen original function bodies detect accidental algorithm changes during moves."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
PREFIX='mimic_baseline.tasks.supermimic'
MAP=json.loads((ROOT/'.github/g1-module-map.json').read_text())


spec=importlib.util.spec_from_file_location('behavior_check',ROOT/'.github/check_behavior.py')
behavior=importlib.util.module_from_spec(spec);spec.loader.exec_module(behavior)


def test_original_algorithm_bodies_are_preserved():
    contract=json.loads((ROOT/'.github/g1-behavior-contract.json').read_text())
    cache={}
    for item in contract['contracts']:
        file=item['file']
        if file not in cache: cache[file]=ast.parse((ROOT/file).read_text())
        node=cache[file]
        for part in item['symbol'].split('.'):
            node=next(n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==part)
        digest=behavior.function_digest(node,MAP)
        assert digest==item['sha256'], (file,item['symbol'])


def test_old_cpu_imports_are_identical_to_canonical_modules():
    for old,new in MAP.items():
        if new.startswith(('core.','data.','learning.','runtime.')):
            assert importlib.import_module(PREFIX+'.'+old) is importlib.import_module(PREFIX+'.'+new)


def test_real_package_import_does_not_load_simulator():
    env=dict(os.environ,PYTHONPATH=str(ROOT/'mimic_baseline/source/mimic_baseline'))
    code='import sys; from mimic_baseline.tasks.supermimic.learning.trainer import Trainer; assert not any(k.startswith(("isaaclab", "omni.")) for k in sys.modules)'
    subprocess.run([sys.executable,'-c',code],env=env,check=True)


def test_public_and_legacy_cli_help_do_not_launch_simulator():
    for base in (ROOT / "scripts", ROOT / "mimic_baseline/scripts/supermimic"):
        for script in ("train_g1", "evaluate_g1", "run_g1_experiment", "export_g1_release"):
            result = subprocess.run([sys.executable, str(base / (script + ".py")), "--help"],
                                    capture_output=True, text=True, check=True)
            assert "usage:" in result.stdout


def test_registered_g1_tasks_use_canonical_manager_environment():
    import gymnasium
    import mimic_baseline.tasks.supermimic
    for task in ("SuperMimic-Flat-G1-v0", "SuperMimic-Recovery-G1-v0"):
        spec = gymnasium.spec(task)
        assert spec.entry_point.endswith("envs.manager_based:SuperMimicRLEnv")
        assert "envs.configs.g1:" in spec.kwargs["env_cfg_entry_point"]


def test_optional_rollout_archive_keeps_original_module_location():
    path = ROOT / "mimic_baseline/source/mimic_baseline/mimic_baseline/tasks/supermimic/evaluation/rollout.py"
    imports = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)]
    archive = next(n for n in imports if any(a.name == "RolloutRecorder" for a in n.names))
    assert archive.level == 0 and archive.module == PREFIX + ".rollout_archive"
