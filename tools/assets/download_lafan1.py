"""Fetch only the G1 LAFAN1 clips, robot description and training ground."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import subprocess
from urllib.request import urlopen

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "g1",
                        help="Asset directory (default: repository data/g1)")
    return parser.parse_args(argv)


def build_jobs(files, data_root):
    clips = [p for p in files if p.parent.as_posix() == "g1" and p.suffix == ".csv"]
    if len(clips) != 40:
        raise RuntimeError(f"Expected 40 G1 LAFAN1 clips, found {len(clips)}")
    jobs = [(f"{HF}/datasets/{DATASET}/resolve/{REVISION}/{p}", data_root / "lafan1_g1" / p.name) for p in clips]
    for path in files:
        rel = path.parts
        if rel[:2] == ("robot_description", "g1"):
            jobs.append((f"{HF}/datasets/{DATASET}/resolve/{REVISION}/{path}", data_root.joinpath("g1_description", *rel[2:])))
    for path in ("default_ground_plane.usda", "Materials/Textures/default_ground_plane_albedo.png",
                 "Materials/Textures/default_ground_plane_roughness.png"):
        jobs.append((f"{GROUND}/{path}", data_root / "ground" / path))
    return jobs


def main(argv=None):
    args = parse_args(argv)
    with urlopen(f"{HF}/api/datasets/{DATASET}/revision/{REVISION}", timeout=30) as response:
        files = [PurePosixPath(item["rfilename"]) for item in json.load(response)["siblings"]]
    jobs = build_jobs(files, args.data_root)
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(download, jobs))
    print("Downloaded 40 G1 LAFAN1 clips and robot/ground assets.")


if __name__ == "__main__":
    main()
