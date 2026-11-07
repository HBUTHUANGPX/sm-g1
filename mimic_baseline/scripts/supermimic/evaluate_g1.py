"""Compatibility entry point for ManagerBased G1 evaluate."""

from pathlib import Path
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "mimic_baseline/source/mimic_baseline")
)
from mimic_baseline.tasks.supermimic.cli.evaluate import main

if __name__ == "__main__":
    main()
