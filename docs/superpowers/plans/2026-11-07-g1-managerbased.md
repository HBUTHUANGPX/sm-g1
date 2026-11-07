# G1 on the original ManagerBased SuperMimic path

Status: implementation and bounded GPU validation complete. No multi-backend work in this task.

## Approved direction
Use the existing `mimic_baseline/tasks/supermimic/env_managed.py`, its manager terms,
reference provider, sampler, replay and Trainer. The separate `supermimic_g1` simulator
is not the new environment implementation. The external repository is only a data-source
reference. Preserve the existing RX-P2 defaults and their tests.

## Implementation sequence
- [x] Audit and parameterize the RX-P2-specific 27-action/29-joint assumptions without changing existing defaults.
- [x] Introduce G1 robot metadata and reuse the original differentiable URDF FK.
- [x] Convert pinned LAFAN1 CSVs into the original MotionLib NPZ schema with explicit frame/quaternion/joint metadata.
- [x] Add G1 configuration to the existing ManagerBased environment, preserving manager lifecycle and terminal transition capture.
- [x] Connect the original SuperMimic Trainer with G1-sized policy, world model and replay; prohibit perception and incompatible checkpoints.
- [x] Run CPU contract tests and existing relevant regressions.
- [x] Run real Windows GPU FK/reset/step/terminal-state checks, short training, resume and full-clip evaluation.
- [x] Publish tested changes immediately on the task branch; report exact evidence and remaining limits.

## Required assertions
G1 has 29 policy actions: its last two joints must not be treated as RX-P2 neck servos.
Joint mappings must be canonical and name-based. Goal features, replay and network dimensions
must agree. Reference velocities use the original convention. Reset must not replace the
physical terminal state in the replay. Real and imagined rewards retain the original MDP.
GPU smoke training is not proof of a converged tracking policy.

## Execution authorization
The user explicitly authorized resuming SSH to the existing Windows machine for environment
checks, short training and evaluation. Do not launch unrelated or indefinite training.

## Publication
The repository remains public. Publish only reviewed source required by this migration,
not historical robot assets, datasets, machine credentials, or company infrastructure.
