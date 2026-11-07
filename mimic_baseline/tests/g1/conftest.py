"""Use the real package initializer, which registers tasks without loading simulation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/mimic_baseline"))
