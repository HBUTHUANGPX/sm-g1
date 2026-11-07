"""Asset CLIs stay usable without simulator dependencies or legacy source."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "tools" / "assets"
NAMES = ["download_lafan1", "urdf_to_mjcf", "mjcf_to_usd"]


@pytest.mark.parametrize("name", NAMES)
def test_help_without_site_packages(name, tmp_path):
    result = subprocess.run([sys.executable, "-S", str(TOOLS / f"{name}.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout


@pytest.mark.parametrize("name", NAMES)
def test_data_root_parsing_without_simulator_imports(name, tmp_path):
    path = TOOLS / f"{name}.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_args([]).data_root == REPO / "data" / "g1"
    assert module.parse_args(["--data-root", str(tmp_path)]).data_root == tmp_path
    assert "supermimic_g1" not in path.read_text()


def test_download_jobs_use_custom_root_and_pinned_g1_only(tmp_path):
    path = TOOLS / "download_lafan1.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("download_lafan1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from pathlib import PurePosixPath
    files = [PurePosixPath(f"g1/clip{i}.csv") for i in range(40)]
    files += [PurePosixPath("other/skip.csv"), PurePosixPath("robot_description/g1/mesh.obj")]
    jobs = module.build_jobs(files, tmp_path)
    assert len(jobs) == 44
    assert all(destination.is_relative_to(tmp_path) for _, destination in jobs)
    assert sum(destination.parent == tmp_path / "lafan1_g1" for _, destination in jobs) == 40
    assert all(module.REVISION in url for url, _ in jobs[:41])
    with pytest.raises(RuntimeError, match="Expected 40"):
        module.build_jobs(files[1:], tmp_path)
