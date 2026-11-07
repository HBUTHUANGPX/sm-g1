# SuperMimic G1

Proprioceptive motion tracking for Unitree G1 on Isaac Lab v3.0.0 using the 40 G1-retargeted LAFAN1 clips. Includes Newton MJWarp and standalone PhysX (OVPhysX), data preparation, offline robot conversion, checkpoint validation and full-clip evaluation.

## Setup

```bash
git clone --recurse-submodules https://github.com/HBUTHUANGPX/sm-g1.git
cd sm-g1
uv sync --project IsaacLab --locked --extra importers --extra ovphysx
cd supermimic_g1
../IsaacLab/.venv/bin/python tools/download_lafan1.py
../IsaacLab/.venv/bin/python tools/urdf_to_mjcf.py
../IsaacLab/.venv/bin/python tools/mjcf_to_usd.py
```

On Windows, use `..\IsaacLab\.venv\Scripts\python.exe`. Install Git, uv and a compatible NVIDIA driver first. Data and generated robot assets are downloaded/generated locally.

See [the training guide](supermimic_g1/README.md) for training, evaluation and resume.

## Results and management

A prior 4,096-environment Newton run completed 40/40 training clips at iteration 13,000 with normalized return 0.8265. Training is stopped. See [the result and evidence limits](docs/results/2026-11-05-newton-lafan1.md).

Every validated independent change is immediately committed and pushed on a task branch. Completed work reaches main through a pull request. See [AGENTS.md](AGENTS.md) and [the workflow](docs/engineering-management.md).

## Attribution

Adapted from [Chong Zhang's supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) at `0525a23a2dcfa61f8543f98b84c075fe66d9d286`, under the [included MIT license](supermimic_g1/LICENSE). This is an unofficial reproduction of SuperMimic.

Isaac Lab is a pinned upstream submodule under its own license. LAFAN1 data and the Unitree description have separate terms in the [dataset card](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset). Data and model checkpoints are not distributed here.
