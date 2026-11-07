"""Lazy compatibility surface for original MDP term imports."""
from importlib import import_module
_target = import_module("mimic_baseline.tasks.supermimic.envs.mdp")
__all__ = _target.__all__
def __getattr__(name):
    return getattr(_target, name)
