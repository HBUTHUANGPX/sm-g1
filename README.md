# SuperMimic G1

The active G1 implementation now extends the original `mimic_baseline` ManagerBased SuperMimic environment and Trainer on Isaac Lab v3.0.0. It uses 29 G1 actions, proprioception only and the 40 G1-retargeted LAFAN1 clips. OVPhysX and Newton/MJWarp environment paths have been verified. The 512-environment Newton 3000-iteration experiment is in progress; its outcome is not yet claimed. Migration to other training platforms remains deferred.

Start with the [ManagerBased G1 guide](docs/managerbased-g1.md). The separate `supermimic_g1/` trainer is retained as a historical implementation; it is not imported by the active training path.

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

The commands above prepare shared robot/data assets. Return to the repository root and follow the [ManagerBased training guide](docs/managerbased-g1.md) for the active train/evaluate/resume entry point.

The [standalone guide](supermimic_g1/README.md) applies only to the historical implementation.

## Results and management

A prior **standalone-implementation** 4,096-environment Newton run completed 40/40 training clips at iteration 13,000 with normalized return 0.8265. This result does not validate the ManagerBased migration or its new policy. See [the result and evidence limits](docs/results/2026-11-05-newton-lafan1.md).

Every validated independent change is immediately committed and pushed on a task branch. Completed work reaches main through a pull request. See [AGENTS.md](AGENTS.md) and [the workflow](docs/engineering-management.md).

## Attribution

The historical `supermimic_g1/` implementation was adapted from [Chong Zhang's supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) at `0525a23a2dcfa61f8543f98b84c075fe66d9d286`, under the [included MIT license](supermimic_g1/LICENSE). This is an unofficial reproduction of SuperMimic.

The active path reuses the original project managers, sampling/reference logic, replay, analytic MDP and BPTT Trainer. Existing upstream license notices are retained in `mimic_baseline/tasks/supermimic` sources; this migration adds G1 metadata and LAFAN1 conversion rather than importing the external trainer.

Isaac Lab is a pinned upstream submodule under its own license. LAFAN1 data and the Unitree description have separate terms in the [dataset card](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset). Data and model checkpoints are not distributed here.
