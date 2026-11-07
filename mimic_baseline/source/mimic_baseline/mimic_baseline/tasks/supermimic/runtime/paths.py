"""Resolve source checkout or exported release without machine-specific paths."""
from pathlib import Path


def project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "mimic_baseline/scripts/supermimic/train_g1.py").is_file():
            return parent
    raise RuntimeError("Expected a source checkout or exported G1 release")


def training_entry():
    return project_root() / "scripts/train_g1.py"
