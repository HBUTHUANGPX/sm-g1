# G1 training execution and memory audit — 2026-11-09

The original ManagerBased environment and Trainer now support a shared batched delayed-PD actuator and two memory-conscious differentiable FK implementations. Training hyperparameters, recovery/reset/randomization/termination behavior, optimizer logic and checkpoint ABI are unchanged. Formal training remains stopped. All training below consists of separate 10-iteration benchmark runs from checkpoint 6600, outside the production checkpoint-selection root.

## Implementation and acceptance

- `GroupedG1DelayedPD` replaces 29 individual controller calls with one tensor operation while retaining independently sampled joint delays, original draw order, simulator joint ordering, history resets, gains and torque limits. The evaluation API remains compatible. Native CUDA differential validation passed **92,800 joint transitions**, including full/partial/empty/slice resets, delay bounds 0 and 4, gain/limit writes, identical torques and identical RNG state.
- `G1LeanKinematics` avoids broadcast copies of static joint transforms and calculates only ancestors of tracked bodies. It retains all four output fields and all six input gradients. It does not detach the policy rollout or reduce its batch/horizon.
- `G1TrainingKinematics` chunks the lean implementation and recomputes deterministic FK during backward. It bounds saved intermediate storage; full input/output tensors still scale with batch size. This is a memory/time tradeoff, not a universally faster implementation.
- The training entry point records execution choices in checkpoint provenance. The profiler supports warmed windows without cProfile, CUDA synchronization and allocator snapshots. The original Trainer's logging, optimizer and cache-clearing cadence is unchanged.
- CLI defaults remain **reference** because end-to-end default-allocator throughput was unstable. Optimized modes are explicitly available; no allocator policy is silently installed. Evaluation defaults remain unchanged.

## Full-shape native CUDA comparison

RTX 5070 Ti, PyTorch 2.11.0+cu130, Windows; source `a320c2342e91af42df49e17c16673b8f3c0170f3`. The isolated benchmark uses the native 29-DoF URDF and **131072 states = 4096 × 32**, two repeats per implementation. Forward plus backward time excludes fixture creation, loss construction and numerical comparisons; peak allocation includes the loss. Each variant gets an isolated graph and benchmark-only cache cleanup.

| FK implementation | Forward + backward, seconds (two runs) | Peak allocated GiB |
|---|---:|---:|
| Original reference | 0.0893 / 0.0868 | 0.872 |
| Previous evaluation batching | 0.1758 / 0.1745 | 1.673 |
| Lean | 0.0818 / 0.0819 | 0.889 |
| Lean + recomputation, chunk 4096 | 0.5857 / 0.5922 | 0.474 |
| Lean + recomputation, chunk 16384 | 0.1631 / 0.1792 | 0.474 |

Thus lean reduces the previous batched FK's peak allocation by about **47%**; bounded lean reduces it by about **72%**. Lean's allocation remains slightly above the original reference. The chunk-4096 execution overhead makes chunk 16384 the preferable measured bounded option on this GPU.

All four outputs and six gradients passed tolerance checks against the reference. Maximum output absolute difference across candidates was **2.6823e-6**, maximum gradient relative L2 error **2.1692e-7**; RNG states were unchanged. These are floating-point equivalence checks, not bitwise identity of entire trained models. Native CPU tests additionally cover double precision gradcheck, MotionLib copy/dtype conversion, arbitrary leading batch dimensions and saved tensor storage.

## Actual training comparison

Every trial preserves 2560 environments, WM batch 2048 × 10, policy batch 4096, horizon 32, replay 1000, warmup 64 and log interval 10. Each starts from the same checkpoint SHA256 `c4d4c8188f91c4683464e64bf7ccb5c4c99047386f906950e5e7e4253705cf51`. There are ten training iterations, five warmup and five measured iterations. The measured window **includes the final ordinary logging and checkpoint I/O**, so it is not a steady-state compute-only rate. GPU trials run serially.

The machine-readable companion records every completed trial, individual iteration times, revisions, allocator settings and memory snapshots. The early `bounded-a` implementation recomputed the older batched FK; `bounded-lean-b` uses the final lean recomputation. One early `grouped-b` trial was deliberately stopped during startup to switch instrumentation; it is excluded. No completed trial is discarded for being slow.

Default allocator timings varied markedly: reference took **72.50 and 10.48 seconds** for five timed iterations, grouped-only took **67.95 seconds**, lean took **62.33 and 63.05 seconds**, and final bounded lean took **50.44 seconds**. These do **not** establish a stable end-to-end speedup. Peak whole-audit allocated/reserved memory was respectively **11.560/14.115 GiB** for reference, **11.586/14.020 GiB** for lean and **11.173/13.641 GiB** for bounded lean.

At iteration boundaries, default-allocator runs had roughly 3.3 GiB of live PyTorch tensors but about 14 GiB reserved, with CUDA reporting zero free device memory. No CUDA allocator OOM/retry events were recorded in the instrumented completed trials. This is evidence of memory pressure, not direct proof of Windows paging or allocator fragmentation as the timing cause.

A process-local diagnostic using `PYTORCH_ALLOC_CONF=per_process_memory_fraction:0.9,garbage_collection_threshold:0.8` reduced reserved memory to about 12 GiB. The same setting was applied to both reference and optimized trials. The complete measurements are below; this remains an explicit experiment, not a production default. A threshold-only trial was fast without reducing reserved memory, and a default-allocator reference trial was also fast: attributing all speed changes to garbage collection would be unsupported. PyTorch documents these options in its [CUDA memory management notes](https://docs.pytorch.org/docs/2.11/notes/cuda.html).

## Reusable commands

The paths below are placeholders for local data/checkpoints and a fresh output directory. These commands are for **bounded validation**, not formal training continuation:

```sh
python scripts/profile_g1_runtime.py \
  --checkpoint <checkpoint-6600> --data-root <data-root> --output <fresh-output> \
  --target training --iterations 10 --warmup-iterations 5 --no-cprofile \
  --actuators grouped --kinematics lean
```

Use `--actuators reference --kinematics reference` for the matched baseline. For lower FK memory, use `--actuators grouped --kinematics bounded --fk-chunk-size 16384`. Do not change allocator settings between members of a paired comparison. The profiler enforces bounded iteration counts and exclusive output directories.

The actual training entry `scripts/train_g1.py` accepts `--training-execution grouped --kinematics-execution lean` or `--kinematics-execution bounded --fk-chunk-size 16384`; explicit `reference` choices restore the old execution. Existing training arguments and resume configuration remain authoritative.

The native FK benchmark is reusable independently of training:

```sh
python -m mimic_baseline.tasks.supermimic.validation.training_kinematics \
  --urdf <native-g1-urdf> --batch 131072 --chunks 4096 16384 --repeats 2 \
  --output <new-report.json>
```

Validation: **170 CPU tests passed, 4 native-runtime skips**; the additional native-URDF CPU FK run passed **20 tests**. Native CUDA actuator/FK checks and the bounded training checks above passed. This audit establishes implementation equivalence and measured execution/memory characteristics; it does not establish long-run reward or tracking-quality equivalence.

## Complete trial ledger

All memory columns are PyTorch whole-audit peaks, including warmup, not total GPU-process memory. Throughput is five divided by the measured window duration. `cap+GC` denotes the explicitly scoped fraction-0.9/threshold-0.8 allocator experiment; `GC only` denotes threshold 0.8 alone.

| Trial | Actuators / FK | Allocator | Seconds / 5 iterations | Iterations/s | Allocated / reserved GiB |
|---|---|---|---:|---:|---:|
| reference-a | reference / reference | default | 72.495 | 0.069 | 11.560 / 14.115 |
| grouped-a | grouped / reference | default | 67.952 | 0.074 | 11.561 / 14.131 |
| bounded-a | grouped / bounded | default | 56.899 | 0.088 | 11.172 / 13.672 |
| lean-a | grouped / lean | default | 62.332 | 0.080 | 11.586 / 14.020 |
| lean-gc08 | grouped / lean | GC only | 9.484 | 0.527 | 11.586 / 14.020 |
| reference-b | reference / reference | default | 10.479 | 0.477 | 11.560 / 14.115 |
| lean-b | grouped / lean | default | 63.053 | 0.079 | 11.586 / 14.020 |
| bounded-lean-b | grouped / bounded | default | 50.438 | 0.099 | 11.173 / 13.641 |
| lean-gc09-08 | grouped / lean | cap+GC | 8.876 | 0.563 | 11.586 / 11.953 |
| reference-gc09-08 | reference / reference | cap+GC | 9.876 | 0.506 | 11.561 / 12.076 |
| lean-gc09-08-repeat | grouped / lean | cap+GC | 79.148 | 0.063 | 11.586 / 11.953 |

**The capped lean repeat took 79.148 seconds despite retaining the same reduced reserved-memory peak.** Neither a stable training speedup nor a cure for the observed runtime variability is established. No automatic allocator/default-execution change is accepted on this evidence.

All **11** completed training checkpoints were loaded on CPU and checked: model tensors finite, model shapes identical to the starting checkpoint, configuration differences limited to the bounded target iteration and separate log directory. Each performed exactly 10 iterations and 310 optimizer steps, with equal transition/window count deltas across implementations. This verifies preserved configuration/update counts, not bitwise equivalence after stochastic simulation. Final process inspection found no remaining Python process on the benchmark machine; the GPU returned to idle. Formal training was not restarted.

The raw paths, model weights, full logs and machine connection details are deliberately excluded. Compact evidence is in [the JSON companion](2026-11-09-g1-training-runtime.json).
