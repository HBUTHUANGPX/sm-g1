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
