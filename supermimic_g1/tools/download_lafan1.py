"""Fetch only the G1 LAFAN1 clips, robot description and training ground."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / "data"
DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
HF = "https://huggingface.co"
GROUND = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.2/Isaac/IsaacLab/Environments/Grid/default_ground_plane_checker_v1"
)


def download(job):
    url, path = job
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--connect-timeout", "15", url, "-o", str(partial)],
        check=True,
    )
    partial.replace(path)


def main():
    with urlopen(f"{HF}/api/datasets/{DATASET}/revision/{REVISION}", timeout=30) as response:
        files = [PurePosixPath(item["rfilename"]) for item in json.load(response)["siblings"]]
    clips = [p for p in files if p.parent.as_posix() == "g1" and p.suffix == ".csv"]
    if len(clips) != 40:
        raise RuntimeError(f"Expected 40 G1 LAFAN1 clips, found {len(clips)}")
    jobs = [(f"{HF}/datasets/{DATASET}/resolve/{REVISION}/{p}", ROOT / "lafan1_g1" / p.name) for p in clips]
    for path in files:
        rel = path.parts
        if rel[:2] == ("robot_description", "g1"):
            jobs.append((f"{HF}/datasets/{DATASET}/resolve/{REVISION}/{path}", ROOT.joinpath("g1_description", *rel[2:])))
    for path in ("default_ground_plane.usda", "Materials/Textures/default_ground_plane_albedo.png",
                 "Materials/Textures/default_ground_plane_roughness.png"):
        jobs.append((f"{GROUND}/{path}", ROOT / "ground" / path))
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(download, jobs))
    print("Downloaded 40 G1 LAFAN1 clips and robot/ground assets.")


if __name__ == "__main__":
    main()
