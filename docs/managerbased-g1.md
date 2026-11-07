# G1 on the original ManagerBased SuperMimic path

## Implementation and scope

New runs default to `SuperMimic-Recovery-G1-v0`; `SuperMimic-Flat-G1-v0` remains available. It instantiates the **original**
`mimic_baseline.tasks.supermimic.envs.manager_based.SuperMimicRLEnv`. The G1 configuration
inherits `SuperMimicEnvCfg`; it does not introduce another physics step loop.
`SuperMimicTrainingEnv` exposes pre-autoreset transitions to the original `Trainer`.
The old standalone package has been removed from the active Git tree. Reusable asset preparation is under `tools/assets/`; existing external data roots remain supported.

Only proprioceptive G1 tracking and LAFAN1 are enabled by the new entry point.
No ERP, PMG, PPO substitution, SOMA or cross-platform abstraction is introduced.
The Isaac Lab 3 runtime supports `--backend ovphysx` (default) and
`--backend newton` (MJWarp solver). Both have prior flat-environment GPU checks; the recovery migration below is verified on Newton.
Resume/evaluation infer the saved backend and reject incompatible overrides.

## What is reused versus changed

| Original component | Treatment |
|---|---|
| `env_managed.SuperMimicRLEnv` and native manager lifecycle | Reused; robot metadata selects torso/actions and suppresses neck handling only for G1. GA full-reset slices are normalized at the environment boundary. |
| Action/command/observation/reward/termination/event/curriculum managers | Same terms and ownership; canonical joint mapping and active-action prefix are robot-specific. |
| `RXP2Kinematics` URDF tree recursion | Reused by `G1Kinematics`; root, joint names and tracked links become constructor parameters. RX-P2 defaults remain. |
| `MotionLib` and `StaticReference` | Same NPZ loader, padding, bins and provider. G1 uses pelvis and all 29 action joints. |
| `MasterySampler` | Reused unchanged. |
| Analytic MDP | Same formulas for real and imagined rewards; observation width follows action width. |
| Replay | Same episode-aware ring and sampling; action fields become configurable. |
| Networks and Trainer | Original architectures, optimization and BPTT; input/output/goal widths account for 29 actions. |
| EpisodeRandomization | Same sampling logic; OVPhysX material view and Newton model-change notification are supported. |
| Evaluator | Original evaluator; G1 selects original unpadded clip endpoints. |
| `g1_robot.py` | New robot configuration from URDF effort/velocity limits, explicit initial PD gains, self-collision enabled. Gains are engineering settings, not hardware calibration. |
| `lafan1.py` | New CSV-to-original-NPZ converter; source digest/schema checks and atomic cache replacement. |

Original RX-P2 has 29 physical joints but only 27 policy actions. G1 has 29 policy
actions: its final wrist joints must participate in actions, references and calibration.
No module-global action dimension is mutated. Old constants/defaults remain for RX-P2.

G1 dimensions: state 73, proprioception 93, reference goal 73 per future frame
(730 for ten frames), optional original relative-orientation feedback 6, context 33.

## Data and assets

Use the existing repository asset preparation commands in the root README.
They download only the pinned G1 LAFAN1 source and robot description and generate
USD. The active entry point consumes these files; it does not use the standalone
trainer, MDP or environment.

Expected `--data-root` structure:

```text
lafan1_g1/*.csv
g1_description/g1_29dof_rev_1_0.urdf
g1_usd/g1_29dof/g1_29dof.usda
ground/default_ground_plane.usda
```

Converted files are local cache under `lafan1_manager_npz/`. CSV quaternion order
is XYZW at 30 Hz; the original NPZ interface uses WXYZ at 50 Hz. The converter
stores root quantities and 29 named joints; the original FK derives tracked bodies.
Source digest and schema version prevent silently reusing a cache after CSV changes.
Large data, USD, logs and checkpoints are not committed.

## Commands

Run from the repository root using the Python environment that contains Isaac Lab
and the selected physics backend. The commands below use `python` as that interpreter. In the validated
Windows installation, its existing OVPhysX dependency directory must remain on
`PYTHONPATH`; do not mix it with another simulator installation.

Prepare / refresh converted data:

```bash
python scripts/train_g1.py --mode prepare --data-root data/g1 --log-dir runs/g1-prepare
```

Flat manager/FK/reward/partial-reset/terminal-transition checks:

```bash
python scripts/train_g1.py --mode check --no-recovery --actuator-type implicit --max-delay 0 --data-root data/g1 --log-dir runs/g1-check --num-envs 4 --clip-limit 1
```

Small-network integration run (64 environments, 10 iterations):

```bash
python scripts/train_g1.py --mode train --data-root data/g1 --log-dir runs/g1-smoke --num-envs 64 --iterations 10 --small-model
```

Original network sizes and 32-step BPTT with small batches:

```bash
python scripts/train_g1.py --mode train --data-root data/g1 --log-dir runs/g1-full-network --num-envs 64 --iterations 2 --buffer-steps 64 --warmup-steps 16 --wm-batches 1 --wm-batch 32 --policy-batch 32 --critic-epochs 1 --critic-minibatches 1
```

The full-network command validates execution, not convergence. Increase environment
and batch sizes only after checking GPU memory; unoverridden optimization settings
are the original Trainer defaults. `--iterations` is an absolute target iteration.

Resume without repeating architecture or training settings:

```bash
python scripts/train_g1.py --mode train --data-root data/g1 --log-dir runs/g1-resume --iterations 12 --resume runs/g1-smoke/last.pt
```

Resume restores saved configuration and infers small/full architecture. Incompatible
training overrides are rejected. Log cadence, target iteration and relocated paths
can change. The original Trainer refills replay after restart; this is not bitwise
continuation of the previous simulator/replay state.

Evaluate all 40 source clips from their first frames with observation/reset noise and physical randomization disabled. Saved actuator delays remain active:

```bash
python scripts/evaluate_g1.py --data-root data/g1 --output runs/g1-eval --checkpoint runs/g1-resume/last.pt
```

The evaluator emits `perclip.json`, `perclip.csv` and `by_category.csv`. Completion
is distinct from a successful evaluation process. New checkpoints use
`supermimic-g1-v1`; old RX-P2 and standalone G1 checkpoints are not accepted as
compatible ManagerBased policies.

## Validation and limits (2026-11-07)

See [machine-readable evidence](results/2026-11-07-managerbased-g1.json).

- Clean deployment from tracked commit `7efd5d2` only: default 10-iteration training
  completed with 64 environments and finite parameters. No training processes remained.
- CPU G1 contracts: 12 passed.
- Original CUDA reference/motion-padding/orientation/randomization regressions:
  64 passed, 2 deselected. The two orientation tests require an unbundled helper or
  use a stale command fixture without `reference_batch`; they are not counted as passing.
- Real GPU manager check, 4 environments: FK position max 8.54e-7 m, rotation
  matrix max 2.22e-6; partial-reset nonselected-state difference zero; real versus
  imagined reward max difference 7.46e-9; terminal states preserved before reset.
- Original Trainer, small networks: 64 environments, 10 iterations; optimizer
  states resumed to 12, then inferred-configuration resume to 13. Parameters finite.
- Original full network sizes and 32-step BPTT: 64 environments, 2 iterations;
  all three optimizers advanced and parameters finite.
- Small-network iteration-12 checkpoint: all 40 source clips evaluated, **0/40
  completed**, mean normalized return about 0.00216. This is a chain-of-execution
  check, not a learned tracking result.

The old standalone Newton 40/40 result must not be attributed to this ManagerBased
migration. No hardware validation has been performed. All training jobs are bounded.

## Newton formal experiment

The initial run was stopped at user request on 2026-11-07. Its latest saved checkpoint is model_2300.pt; the automatic final evaluation was cancelled. This is not a completed 3000-iteration or evaluated tracking result. The following commands document that initial run. It uses the
original full networks and optimization settings (WM batch 2048, 10 WM updates,
policy batch 4096, 32-step horizon, replay 1000 steps, seed 0). A 512-environment
capacity trial passed three iterations at approximately 0.53 iterations/s after
warmup, with 9.54 GiB peak PyTorch allocation. Physics uses 5 ms steps and the
original manager decimation of four; measured action time is 20 ms.

```bash
python scripts/train_g1.py --backend newton --mode train --data-root data/g1 --log-dir runs/newton-512-3000 --num-envs 512 --iterations 3000 --save-every 100 --log-every 10
python scripts/train_g1.py --mode evaluate --data-root data/g1 --log-dir runs/newton-512-3000-eval --resume runs/newton-512-3000/model_3000.pt
```

These are historical commands for pinned revision `282a555`, not commands to restart the stopped experiment on current defaults. The second command was not run; final evaluation was cancelled. Evaluation includes all
40 source clips and uses their original unpadded endpoints. Newton validation
checks actual MJWarp geometry friction after model updates, the ground coefficient,
COM rebasing, partial resets, reward parity and pre-autoreset terminal states.
See [Newton evidence](results/2026-11-07-managerbased-g1-newton.json).

The formal run is pinned to source revision `282a555`. Its disabled recovery
provenance contains legacy descriptive action/observation widths 27/91; actual
policy, replay and observations are verified as G1 29/93. The subsequent metadata
fix affects future checkpoints only. It does not change learning or this run.


## G1 recovery migration (2026-11-07)

New runs enable the original RX-P2 recovery lifecycle. The stopped flat run remains
at its last saved iteration 2300 and is not resumed as a recovery experiment.
The separate recovery deployment contains only bounded validation jobs.

| Feature | Current G1 behavior |
|---|---|
| Action delay | Original reset-safe ring buffers, independently sampled 0–4 physics ticks (0–20 ms), explicit PD, 5 ms physics tick / 20 ms control tick. Zero-time resets do not advance history. |
| Newton | Lab computes delayed PD; Newton MJWarp integrates physics. Self-collision is explicitly enabled; actual robot-to-robot contacts were observed. |
| Large-pose reset | Original 15% recovery role, reference yaw with roll/pitch uniform in [-pi, pi], reference joints ±0.2 rad clipped to limits, zero velocities. Actual G1 collision boxes start 0.15 m above ground. Instanced USD meshes are included. |
| Tracking termination | Original height / tilt / ankle / wrist groups. Independent probabilities per 20 ms step: 0.006667 / 0.006667 / 0.04 / 0.04, combined by OR. |
| Protection | Recovery-role episodes only, first 200 control steps (4 s); motion-end and time-limit truncations remain active. Evaluation uses deterministic raw tracking failures without protection. |
| Imagined training | Original smooth group-OR expected hazard and source-state protection clock; replay records recovery role, critic receives remaining protection. |
| Dimensions | Action 29, state 73, actor observation 93, 10-frame goal 730, WM 162→73, recovery critic 915. Original RX-P2 defaults remain unchanged. |
| TN | Not enabled. Static URDF effort limits remain active. |

Explicit PD requires a reflected-inertia setting: delayed G1 actuators now use
**0.03 kg·m² armature per joint**, while implicit legacy configurations are unchanged.
This is an engineering simulation setting, not measured G1 motor identification.
Without it the tested explicit-PD configuration diverged at the fourth control
step; with it the bounded reset/rollout and full-network training checks passed.
No claim is made about long-run convergence or hardware fidelity.

Run the recovery environment check (at least 32 environments):

```bash
python scripts/train_g1.py --backend newton --mode check --recovery --num-envs 64 --clip-limit 1 --data-root data/g1 --log-dir runs/g1-recovery-check
```

Reproduce the full-network integration check with small training batches:

```bash
python scripts/train_g1.py --backend newton --mode train --recovery --num-envs 64 --iterations 10 --buffer-steps 128 --warmup-steps 64 --wm-batch 32 --wm-batches 1 --policy-batch 32 --critic-epochs 1 --critic-minibatches 1 --data-root data/g1 --log-dir runs/g1-recovery-smoke
python scripts/train_g1.py --mode train --iterations 12 --resume runs/g1-recovery-smoke/last.pt --data-root data/g1 --log-dir runs/g1-recovery-resume
```

Resume restores recovery/actuator/collision settings; incompatible overrides fail.
Old Newton checkpoints missing a self-collision field restore their previous
`False` setting. `--no-recovery` selects flat termination/reset semantics;
`--actuator-type implicit --max-delay 0` independently selects implicit actuation.

See [recovery validation evidence](results/2026-11-07-managerbased-g1-recovery.json)
for measured reset rates, clearance, protection boundaries, self-contact and
checkpoint hashes. These checks validate execution and semantics, not learned recovery.

The iteration-12 smoke checkpoint was evaluated across all 40 LAFAN1 clips: 0/40 completed. This is an evaluation-entry check, not evidence of learned recovery. No training process remains running.

## Pinned train-then-evaluate jobs

Export a release from a committed revision on the development machine:

```bash
python mimic_baseline/scripts/supermimic/export_g1_release.py --output g1-release.tar
```

Extract it into a new deployment directory on the training machine. The archive
contains only committed publication-allowlisted files and `release-manifest.json`.
Use the printed full source revision with the Isaac Lab Python interpreter:

```bash
python mimic_baseline/scripts/supermimic/run_g1_experiment.py --data-root DATA_ROOT --run-dir NEW_RUN_DIRECTORY --num-envs VALIDATED_ENV_COUNT --iterations 3000 --source-revision FULL_SOURCE_SHA
```

The runner refuses an existing run directory, verifies deployed source hashes
before training and evaluation, saves checkpoints every 100 iterations, and only
runs evaluation after a successful final checkpoint with finite model parameters.
It requires exactly the 40 source clip names in the evaluation output; completion
of the pipeline does not mean completion of every motion. `status.json` records
training/evaluation failure or completion, while `experiment.json` preserves the
commands and source manifest. Runtime logs and checkpoints remain outside Git.


## Fresh recovery formal run

The new Newton recovery run started on 2026-11-07 with **2560 environments**,
3000 target iterations, full networks, WM batch2048×10, policy batch4096,
32-step BPTT and 1000-step replay. It starts from scratch rather than the stopped
flat checkpoint. Source is pinned to `1bf1a4e9d788316046bad25bb97ba2018fe0cc01`.
The Windows process is detached from SSH and invokes all-40-clip evaluation only
after verified successful training. Saving occurs every 100 iterations.

Capacity trials covered 1024, 2048, 2560, 3072 and 4096 environments. The selected
2560 configuration completed a 60-iteration full-batch stability run at about
0.54–0.56 iterations/s, with peak PyTorch allocation11.57 GiB and reservation14.22 GiB.
Larger counts incurred severe throughput degradation; the choice favors practical
capacity with GPU headroom. The 60-iteration smoke model completed0/40 clips,
which does not predict the formal training outcome. Formal results are pending.

See [capacity evidence and launch snapshot](results/2026-11-07-g1-recovery-formal.json).


## Source layout

See [architecture and evaluation](architecture.md) for canonical modules, stable command wrappers, compatibility contracts and release validation.


## Recovery 3000-iteration result (2026-11-07)

The separate Newton recovery run completed 3000 iterations with 2560 environments
on pinned source `1bf1a4e9d788316046bad25bb97ba2018fe0cc01`. The automatic
evaluator processed all 40 LAFAN1 clips: **0/40 completed**, mean normalized return
**0.06922115**. All ended in native tracking termination. This is a completed
experiment with unsuccessful full-clip tracking, not evidence of convergence.
The source architecture refactor was not deployed into this run.

See [all 40 per-clip results](results/2026-11-07-g1-recovery-lafan1-3000.json)
for durations, error metrics, outcomes and checkpoint SHA256. No training settings
were changed in response to this result as part of the architecture task.


## Continuing recovery to 20000 iterations

The user authorized continuing the completed recovery run from iteration 3000
to **total iteration 20000**, with the same 2560 environments, Newton physics,
full networks, recovery rules and optimizer settings. The launcher now accepts
`--resume`, records the input iteration/SHA256, checks configuration compatibility
and optimizer presence, and rechecks the input digest before launching.

In a committed, exported release, use the release manifest's exact source SHA:

```bash
python scripts/run_g1_experiment.py --data-root supermimic_g1/data --run-dir runs/newton-recovery-2560-20000 --num-envs 2560 --iterations 20000 --resume runs/newton-recovery-2560-3000/train/model_3000.pt --source-revision 61cdbfe36f1eb08d7294f5214ee25c204874762a
```

Paths may point to separate deployments. Existing output directories are rejected.
Checkpoints remain every 100 iterations and metrics every 10; all-40 evaluation
runs automatically after a verified final checkpoint. Resume restores model and
optimizer state, then refills replay using the existing Trainer behavior.
The original 3000-iteration evaluation remains intact.

See [launch metadata](results/2026-11-07-g1-recovery-20000.json). This records a
running continuation, not a completed 20000-iteration result. Source, runtime and
data are pinned; no learning or physics algorithm was changed for continuation.
