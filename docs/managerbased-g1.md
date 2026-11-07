# G1 on the original ManagerBased SuperMimic path

## Implementation and scope

New runs default to `SuperMimic-Recovery-G1-v0`; `SuperMimic-Flat-G1-v0` remains available. It instantiates the **original**
`mimic_baseline.tasks.supermimic.env_managed.SuperMimicRLEnv`. The G1 configuration
inherits `SuperMimicEnvCfg`; it does not introduce another physics step loop.
`SuperMimicTrainingEnv` exposes pre-autoreset transitions to the original `Trainer`.
The standalone `supermimic_g1/supermimic` package is not imported.

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
python mimic_baseline/scripts/supermimic/train_g1.py --mode prepare --data-root supermimic_g1/data --log-dir runs/g1-prepare
```

Flat manager/FK/reward/partial-reset/terminal-transition checks:

```bash
python mimic_baseline/scripts/supermimic/train_g1.py --mode check --no-recovery --actuator-type implicit --max-delay 0 --data-root supermimic_g1/data --log-dir runs/g1-check --num-envs 4 --clip-limit 1
```

Small-network integration run (64 environments, 10 iterations):

```bash
python mimic_baseline/scripts/supermimic/train_g1.py --mode train --data-root supermimic_g1/data --log-dir runs/g1-smoke --num-envs 64 --iterations 10 --small-model
```

Original network sizes and 32-step BPTT with small batches:

```bash
python mimic_baseline/scripts/supermimic/train_g1.py --mode train --data-root supermimic_g1/data --log-dir runs/g1-full-network --num-envs 64 --iterations 2 --buffer-steps 64 --warmup-steps 16 --wm-batches 1 --wm-batch 32 --policy-batch 32 --critic-epochs 1 --critic-minibatches 1
```

The full-network command validates execution, not convergence. Increase environment
and batch sizes only after checking GPU memory; unoverridden optimization settings
are the original Trainer defaults. `--iterations` is an absolute target iteration.

Resume without repeating architecture or training settings:

```bash
python mimic_baseline/scripts/supermimic/train_g1.py --mode train --data-root supermimic_g1/data --log-dir runs/g1-resume --iterations 12 --resume runs/g1-smoke/last.pt
```

Resume restores saved configuration and infers small/full architecture. Incompatible
training overrides are rejected. Log cadence, target iteration and relocated paths
can change. The original Trainer refills replay after restart; this is not bitwise
continuation of the previous simulator/replay state.

Evaluate every source clip from its first frame with randomization/noise disabled:

```bash
python mimic_baseline/scripts/supermimic/train_g1.py --mode evaluate --data-root supermimic_g1/data --log-dir runs/g1-eval --resume runs/g1-resume/last.pt
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
python mimic_baseline/scripts/supermimic/train_g1.py --backend newton --mode train --data-root supermimic_g1/data --log-dir runs/newton-512-3000 --num-envs 512 --iterations 3000 --save-every 100 --log-every 10
python mimic_baseline/scripts/supermimic/train_g1.py --mode evaluate --data-root supermimic_g1/data --log-dir runs/newton-512-3000-eval --resume runs/newton-512-3000/model_3000.pt
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
python mimic_baseline/scripts/supermimic/train_g1.py --backend newton --mode check --recovery --num-envs 64 --clip-limit 1 --data-root supermimic_g1/data --log-dir runs/g1-recovery-check
```

Reproduce the full-network integration check with small training batches:

```bash
python mimic_baseline/scripts/supermimic/train_g1.py --backend newton --mode train --recovery --num-envs 64 --iterations 10 --buffer-steps 128 --warmup-steps 64 --wm-batch 32 --wm-batches 1 --policy-batch 32 --critic-epochs 1 --critic-minibatches 1 --data-root supermimic_g1/data --log-dir runs/g1-recovery-smoke
python mimic_baseline/scripts/supermimic/train_g1.py --mode train --iterations 12 --resume runs/g1-recovery-smoke/last.pt --data-root supermimic_g1/data --log-dir runs/g1-recovery-resume
```

Resume restores recovery/actuator/collision settings; incompatible overrides fail.
Old Newton checkpoints missing a self-collision field restore their previous
`False` setting. `--no-recovery` selects flat termination/reset semantics;
`--actuator-type implicit --max-delay 0` independently selects implicit actuation.

See [recovery validation evidence](results/2026-11-07-managerbased-g1-recovery.json)
for measured reset rates, clearance, protection boundaries, self-contact and
checkpoint hashes. These checks validate execution and semantics, not learned recovery.

The iteration-12 smoke checkpoint was evaluated across all 40 LAFAN1 clips: 0/40 completed. This is an evaluation-entry check, not evidence of learned recovery. No training process remains running.
