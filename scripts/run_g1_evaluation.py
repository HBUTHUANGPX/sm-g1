"""Public entry point for a complete ManagerBased G1 evaluation job."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "mimic_baseline/source/mimic_baseline")
)
from mimic_baseline.tasks.supermimic.cli.evaluation_job import main

if __name__ == "__main__":
    main()
