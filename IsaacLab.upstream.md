# Isaac Lab baseline

- Upstream: https://github.com/isaac-sim/IsaacLab.git
- Tag: `v3.0.0`
- Commit: `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`
- Managed as the `IsaacLab/` submodule. Initialize with `git submodule update --init`.
- Python 3.12, PyTorch 2.11/CUDA 13.0 and backend dependencies use upstream `uv.lock`.

Project adaptations live in `supermimic_g1/`: GA launcher and physics manager APIs, named joint mapping, indexed state writes, and backend-specific configuration. Reference implementation baseline: `zita-ch/supermimic_vibed@0525a23a2dcfa61f8543f98b84c075fe66d9d286`.
