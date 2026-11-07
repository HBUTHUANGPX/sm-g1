"""Public entry point for queued post-run G1 evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mimic_baseline/source/mimic_baseline'))
from mimic_baseline.tasks.supermimic.cli.evaluate_after_run import main

if __name__ == '__main__':
    main()
