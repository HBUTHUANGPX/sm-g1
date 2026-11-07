"""Load CPU math modules without launching or importing the simulator."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'source/mimic_baseline/mimic_baseline'
for name, path in [('mimic_baseline', ROOT), ('mimic_baseline.tasks', ROOT / 'tasks'),
                   ('mimic_baseline.tasks.supermimic', ROOT / 'tasks/supermimic')]:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
