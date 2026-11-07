# Unitree G1 SuperMimic on Isaac Lab v3.0.0

This is the project's G1-only training path. It uses proprioception and reference motion, with no camera or terrain perception. The trainer collects simulated rollouts, fits a differentiable world model, and optimizes the actor through that model with a critic bootstrap.

The implementation is adapted from [Chong Zhang's SuperMimic reproduction](https://github.com/zita-ch/supermimic_vibed/tree/0525a23a2dcfa61f8543f98b84c075fe66d9d286), under the included MIT license. The runtime is the project's official Isaac Lab v3.0.0, rather than the reference repository's development commit.

## Assets and dataset

`tools/download_lafan1.py` fetches only the 40 29-DoF G1-retargeted LAFAN1 CSVs and their matching Unitree robot description at a pinned Hugging Face revision, plus the Isaac Lab ground asset. The 264,705 source frames are at 30 Hz. The motion loader resamples them to 50 Hz, retaining the CSV's XYZW input convention and computing tracked-body state from the matching G1 kinematics. The training dataset does not include SOMA.

The offline converters build the free-base MJCF from the URDF and use the Isaac Sim 6.2 standalone MJCF importer to produce a layered USD with `mujoco` and `physx` variants. Keep the whole generated USD directory with its payloads. `data/` is local and ignored. See the [dataset card](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) for attribution and terms.

## Setup on a fresh NVIDIA machine

From the parent project, use the bundled GA lock:

```bash
uv sync --project IsaacLab --locked --extra importers --extra ovphysx
```

Then use that environment's Python from `supermimic_g1/`:

```bash
../IsaacLab/.venv/bin/python tools/download_lafan1.py
../IsaacLab/.venv/bin/python tools/urdf_to_mjcf.py
../IsaacLab/.venv/bin/python tools/mjcf_to_usd.py
```

On Windows, use the Python executable at the parent project's IsaacLab/.venv/Scripts/python.exe. Run conversion in a separate process before training.

## 3,000-iteration runs

On Windows, from this directory:

```cmd
train_3000.cmd newton
train_3000.cmd ovphysx
```

The launchers use the same 64 environments, seed, dataset, optimizer schedule, 50 Hz control and 200 Hz physics. The two runs use distinct output directories. The 3,000-iteration profile uses 2 world-model minibatches of 32 windows, an 8-step imagined rollout with 32 starts, 100 warm-up iterations for the world model, and the reference model widths. This is a compact validation profile, not a reproduction of the reference repository's full-scale benchmark. Domain randomization retains joint offsets and torso CoM offsets. The Newton material coefficient is fixed to 1.0 and PhysX uses the authored/default materials. Their contact behavior is not expected to match exactly.

On Linux or for a custom run, use the active Isaac Lab Python:

```bash
python supermimic/train.py --backend newton --device cuda:0 --viz none \
  --iterations 3000 --num_envs 64 --buffer_steps 256 --warmup_steps 64 \
  --wm_batches 2 --wm_batch 32 --pi_batch 32 --horizon 8 --wm_only_iters 100 \
  --critic_epochs 1 --ckpt_every 500 --log_every 25 --lr_decay_iters 3000 \
  --log_dir logs/newton-3000
```

For standalone PhysX, use `--backend ovphysx --log_dir logs/ovphysx-3000`. The separate `isaacsim_physx` option requires the full Isaac Sim 6.2 runtime and is not part of the Windows validation profile. Windows defaults to eager PyTorch, and Linux can use `--no-compile` if a compiler is unavailable.

Training writes `train_log.csv`, `last.pt` (including optimizers), and numbered policy/world-model checkpoints. To resume, rerun the same training command with the same backend and options and add `--resume logs/<run>/last.pt`.

To evaluate a checkpoint on the 40 training clips:

```bash
python supermimic/evaluate.py --backend newton --viz none \
  --run_dir logs/newton-3000 --ckpt logs/newton-3000/ckpt_003000.pt
```

## Verified Windows runs

Both runs completed 3,000 iterations on the RTX 5070 Ti on 2026-11-05 and collected 484,096 simulation transitions each. `tools/verify_run.py` checked finite log metrics and weights, loaded both final checkpoints, and confirmed optimizer state was present. The numbered checkpoints and logs are copied to this Mac project's `logs/newton-3000/` and `logs/ovphysx-3000/`. The Windows run directories also contain `last.pt` for resuming.

| Backend | Iterations | Normalized full-clip return | Clips terminated early |
| --- | ---: | ---: | ---: |
| Newton MJWarp | 3,000 | 0.002064 | 40/40 |
| OVPhysX | 3,000 | 0.001425 | 40/40 |

This run validates the training and checkpoint path. It does not demonstrate a converged tracker.

## Larger Newton LAFAN1 run

`train_newton_lafan1.cmd` starts a fresh 4,096-environment Newton run with a 20,000-iteration budget. It uses 512 replay steps per environment, eight world-model updates of 1,024 windows per iteration, 1,024 imagined starts with a 32-step horizon, and 500 world-model warm-up iterations. It retains the longer reference learning-rate schedule rather than decaying to the minimum by iteration 3,000.

The original validation used Windows tasks named `SuperMimic-G1-LAFAN1-Newton-4096` (training) and `SuperMimic-G1-LAFAN1-Newton-Eval` (evaluation). A new machine needs its own scheduler setup if background execution is desired. The run writes `logs/newton-lafan1-4096/train_log.csv` and a checkpoint every 500 iterations. `tools/watch_lafan1.py` evaluates every 1,000th iteration from the first frame of each of the 40 clips. It writes one report per evaluated checkpoint and `best_eval.json`, ranked first by clips completed, then by normalized return.

Read the latest row of logs/newton-lafan1-4096/train_log.csv to inspect training. Read best_eval.json after an evaluation to inspect full-clip tracking. A high reward from sampled reset frames alone does not establish full-clip tracking.
