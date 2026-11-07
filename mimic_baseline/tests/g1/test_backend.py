import pytest
from mimic_baseline.tasks.supermimic.g1_run_config import resolve_backend


def test_backend_defaults_and_resume_identity():
    assert resolve_backend(None, None) == "ovphysx"
    assert resolve_backend("newton", None) == "newton"
    old = {"provenance": {}}
    assert resolve_backend(None, old) == "ovphysx"
    new = {"provenance": {"backend": "newton"}}
    assert resolve_backend(None, new) == "newton"
    with pytest.raises(ValueError, match="backend"):
        resolve_backend("ovphysx", new)
    with pytest.raises(ValueError, match="backend"):
        resolve_backend("other", None)
