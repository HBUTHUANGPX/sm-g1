"""Compatibility alias; canonical implementation: envs.legacy."""
from importlib import import_module as _import_module
import sys as _sys
_target = _import_module("mimic_baseline.tasks.supermimic.envs.legacy")
globals().update({name: value for name, value in vars(_target).items() if not name.startswith("__")})
_sys.modules[__name__] = _target
