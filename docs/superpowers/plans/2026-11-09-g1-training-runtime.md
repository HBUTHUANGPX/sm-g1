# G1 Training Runtime Implementation Plan

> For agentic workers: use superpowers:executing-plans to execute the checks below, with independent actuator/FK investigations coordinated by the root agent.

**Goal:** Batch G1 training actuators and bound differentiable FK memory without changing training samples, physics, losses or update counts.

**Architecture:** Reuse the already differential-tested independent-joint delay implementation through a shared actuator API. Add an opt-in chunked/recomputed FK alongside both reference and evaluation FK; select production defaults only after full-shape GPU acceptance.

**Tech Stack:** Original ManagerBased environment/Trainer, Isaac Lab v3.0.0, Newton, PyTorch CUDA.

**Spec:** Current user request: original training configuration and semantics unchanged; compare throughput, memory and numerics; commit/push validated changes; do not resume formal training.

## Global constraints

- Keep 2560 environments, 4096 policy batch, horizon 32, WM batch 2048 with 10 updates, replay 1000 and warmup 64; preserve logging/random-number consumption.
- Preserve per-joint independent delays, joint order, gains/limits, reset history and RNG order.
- Keep full six-input FK gradients. No detached BPTT, smaller training batch, reduced collision checks, altered recovery/reset/termination or TN changes.
- Run bounded trials into separate directories from checkpoint 6600. Formal training remains stopped.
- Keep checkpoints/logs/machine details private. Publish only code and compact evidence.

## Approach and review focus

Reference per-link FK is the numerical baseline. Evaluation depth batching retains more native-URDF backward storage; unrestricted training batching is not accepted. Bound temporary FK work by chunking and recomputing its deterministic graph during backward. This trades recomputation for memory; GPU evidence decides chunk size and whether the candidate is useful. Do not claim a cause from device-memory samples alone.

Review empty/partial/slice resets, simulator joint-order permutations, gain writes, MotionLib shallow-copy double transforms, all input gradients, RNG preservation, native full-shape peak allocation and timing variability.

## Tasks

- [x] Shared actuator API in envs/grouped_actuators.py; retain evaluation aliases. Test exact torque/parameters/delay/RNG with native CUDA validator before choosing training default. Commit/push independently.
- [x] G1TrainingKinematics in robots/g1/kinematics.py. Test values, all six gradients, float64 gradcheck, batch shapes, copy/dtype and saved storage; measure full 131072-state CUDA forward/backward at candidate chunk sizes. Commit/push independently.
- [x] Extend supported training CLI/profiler with recorded execution choices and bounded warmup/no-cProfile comparison. Keep TrainingConfig/checkpoint ABI and original Trainer methods intact.
- [x] Compare reference, grouped-only and grouped+bounded-FK under the same checkpoint/configuration, with repeat trials, peak allocated/reserved memory, finite parameters and numerical evidence. Do not infer training improvement from simulation nondeterminism.
- [x] Retain reference CLI defaults: native equivalence passed, but default-allocator end-to-end throughput remains unstable. Publish explicit optimized execution choices and measurements; complete CPU/CI and PR integration.
