"""Export only the publication allowlist from a Git commit, with SHA256 checksums."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from ..runtime.paths import project_root


def export_release(repo, output, revision="HEAD"):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    revision = git("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    paths = (
        git("show", revision + ":.github/mimic-publication-files.txt")
        .decode()
        .splitlines()
    )
    files = {}
    manifest = {"source_revision": revision, "files": files}
    # Exclusive creation prevents replacing an archive used by a running job.
    with (
        open(output, "xb") as handle,
        tarfile.open(fileobj=handle, mode="w") as archive,
    ):
        for name in paths:
            if not name or name.startswith("#"):
                continue
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid publication path")
            data = git("show", revision + ":" + name)
            files[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("release-manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    result = export_release(project_root(), args.output, args.revision)
    print(
        json.dumps(
            {
                "source_revision": result["source_revision"],
                "files": len(result["files"]),
            }
        )
    )
