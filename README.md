# SuperMimic G1

The active G1 implementation now extends the original `mimic_baseline` ManagerBased SuperMimic environment and Trainer on Isaac Lab v3.0.0. It uses 29 G1 actions, proprioception only and the 40 G1-retargeted LAFAN1 clips. OVPhysX and Newton/MJWarp environment paths have been verified. The initial 512-environment Newton experiment was stopped at user request; its latest saved checkpoint is iteration 2300 and no final evaluation is claimed. The subsequent 2560-environment recovery run completed 3000 iterations and evaluated all 40 clips: 0/40 completed (mean normalized return 0.06922). Migration to other training platforms remains deferred.

Start with the [ManagerBased G1 guide](docs/managerbased-g1.md) and [source architecture / all-40 evaluation guide](docs/architecture.md). The original ManagerBased path is the sole active implementation. The removed standalone trainer remains available in [Git history at `4727620`](https://github.com/HBUTHUANGPX/sm-g1/tree/4727620b31bf3af0bf43adcda1c27e2c815740e1/supermimic_g1). See the [repository layout](docs/repository-layout.md) and [strict / full-clean evaluation protocols](docs/evaluation-profiles.md).

## Setup

```bash
git clone --recurse-submodules https://github.com/HBUTHUANGPX/sm-g1.git
cd sm-g1
uv sync --project IsaacLab --locked --extra importers --extra ovphysx
IsaacLab/.venv/bin/python tools/assets/download_lafan1.py
IsaacLab/.venv/bin/python tools/assets/urdf_to_mjcf.py
IsaacLab/.venv/bin/python tools/assets/mjcf_to_usd.py
```

On Windows, use `IsaacLab\.venv\Scripts\python.exe`. Install Git, uv and a compatible NVIDIA driver first. Data and generated robot assets are downloaded/generated locally.

The commands above prepare shared robot/data assets under `data/g1/`. Each tool accepts `--data-root PATH` for an existing or alternate asset directory and `--help` without simulator dependencies. Follow the [ManagerBased training guide](docs/managerbased-g1.md) for the active train/evaluate/resume entry point.

Existing local datasets and checkpoints are preserved; pass their existing directory with `--data-root` when needed.

## Results and management

A prior **standalone-implementation** 4,096-environment Newton run completed 40/40 training clips at iteration 13,000 with normalized return 0.8265. This result does not validate the ManagerBased migration or its new policy. See [the result and evidence limits](docs/results/2026-11-05-newton-lafan1.md).

Every validated independent change is immediately committed and pushed on a task branch. Completed work reaches main through a pull request. See [AGENTS.md](AGENTS.md) and [the workflow](docs/engineering-management.md).

## Attribution

The historical `supermimic_g1/` implementation was adapted from [Chong Zhang's supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) at `0525a23a2dcfa61f8543f98b84c075fe66d9d286`, under the [included MIT license](licenses/supermimic_vibed-MIT.txt). This is an unofficial reproduction of SuperMimic.

The active path reuses the original project managers, sampling/reference logic, replay, analytic MDP and BPTT Trainer. Existing upstream license notices are retained in `mimic_baseline/tasks/supermimic` sources; this migration adds G1 metadata and LAFAN1 conversion rather than importing the external trainer.

Isaac Lab is a pinned upstream submodule under its own license. LAFAN1 data and the Unitree description have separate terms in the [dataset card](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset). Data and model checkpoints are not distributed here.
