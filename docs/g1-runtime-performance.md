# G1 runtime performance audit

This audit covers the original ManagerBased G1 recovery path with Newton, proprioception and LAFAN1. It separates short-run observations from performance hypotheses. Benchmark checkpoints and full traces remain local; this document contains no machine connection information.

## Origin of the avoidable overhead

| Area | Origin and finding | Disposition |
| --- | --- | --- |
| 29 separate delayed-PD actuators | Introduced by the G1 adaptation. Independent joint delays did not require 29 Python actuator dispatches. This should have been profiled before formal runs. | Evaluation uses one batched actuator while preserving independent delays, joint order and reset RNG. Training is measured separately; its production default is retained. |
| Per-link differentiable FK | Inherited original tree recursion, reused for G1 without measuring the repeated small CUDA launches. | Optional depth batching is validated for evaluation; full-shape training is not accepted. |
| Repeated canonical-state reads | Original ManagerBased observations read state independently; existing post-state/reward/FK caches already work. | Evaluation alone reuses the post-physics snapshot inside the same no-autoreset step. Training reset boundaries remain unchanged. |
| Two evaluation protocols scheduled serially | Evaluation orchestration choice, not a mathematical requirement. | Isolated processes support concurrent execution; choose scheduling from complete-job wall time and memory measurements. |
| BPTT, model updates, replay, diagnostics | Original training work, not interchangeable with inference. Diagnostics also consume RNG. | Profile and retain semantics; do not shorten horizons, skip updates/checks or change logging cadence to manufacture a speedup. |

## Bounded training comparison

Both completed profiles resumed iteration 6600 from checkpoint SHA-256 `c4d4c8188f91c4683464e64bf7ccb5c4c99047386f906950e5e7e4253705cf51`. Their verified source revision was `1eb4899730d8c03258c88d0f7f236e36efd1968d`, manifest SHA-256 `e439019b7fb4ed92bb64cea8d2fcef515d79a721c0edc4fe0c83f6d5d60d3657`.

The benchmark retained 2560 environments, 2048 world-model windows per minibatch, 10 world-model minibatches per iteration, policy batch 4096, horizon 32, replay capacity 1000 steps, warmup 64 steps and logging interval 10. Both used reference FK. The grouped actuator variant was enabled only for this benchmark; these results do not authorize changing a formal training run.

| Observation | Reference actuators | Grouped actuators |
| --- | ---: | ---: |
| Completed measured iterations | 5 | 5 |
| Synchronized measured interval | 20.3823 s | 9.9510 s |
| Replay refill, measured separately | 6.2341 s | 4.4868 s |
| Peak Torch allocated memory | 11.5597 GiB | 11.5587 GiB |
| Peak Torch reserved memory | 14.1133 GiB | 14.1309 GiB |
| World-model update calls | 50 | 50 |
| Policy/critic update calls | 5 | 5 |
| Backward calls | 155 | 155 |
| Simulator control steps / physics steps | 12 / 48 | 12 / 48 |
| Log calls / environment reset calls | 1 / 8 | 1 / 8 |

A repeated reference-actuator/reference-FK run subsequently completed five iterations in **78.2006 s**, with refill 6.2174 s and the same peak allocated/reserved memory (11.5597/14.1133 GiB). Its device-monitor sample showed 15964 MiB used, 100% GPU utilization and approximately 81 W. Thus even the unchanged reference configuration varied from 20.3823 s to 78.2006 s. Further grouped/reference repetitions are pending.

These are short, instrumented runs with unstable training throughput near device capacity, not a demonstrated twofold steady-state speedup. Timing starts after replay refill and synchronizes at both measurement boundaries. The normal Trainer lifecycle remains active, including final logging and checkpoint behavior. Startup effects, allocator/kernel warmup, execution order and cProfile overhead have not been separated by repeated alternating trials. Torch allocation statistics also exclude memory owned by the simulator and other native allocations.

A further alternating trial completed reference actuators in **10.5375 s** (refill 9.6735 s) and grouped actuators in **10.3759 s** (refill 4.5237 s). Across completed trials the reference range is **10.54–78.20 s** and grouped range **9.95–10.38 s**. Their fastest runs are close; these measurements do not support transferring the large evaluation speedup to training. All completed training trials reported finite model parameters, retained original batch/update counts and wrote only separate benchmark outputs.

## Hotspot comparison and attribution limits

The table reports **cProfile cumulative host-observed seconds**, not CUDA kernel time. Rows are nested and must not be added together. GPU execution is asynchronous: a later operation may wait for earlier queued work, and allocator or implicit synchronization can move apparent time between functions. Changes in unchanged Adam, replay and FK functions below are therefore not evidence that their algorithms became faster.

| Function or stage | Reference | Grouped | Interpretation |
| --- | ---: | ---: | --- |
| World-model updates | 9.8111 | 3.2623 | Largest reference host-observed stage; requires device timeline attribution. |
| Actor and critic updates | 7.1032 | 5.2363 | Largest grouped stage; includes BPTT and gradient work. |
| Backward | 5.0727 | 3.1209 | Same 155 calls; not an independent additive stage. |
| Adam step | 3.4702 | 0.2039 | Same 155 calls; large variation is a warning against causal per-function attribution. |
| Collection | 2.6234 | 1.1374 | Directly contains the actuator execution change. |
| Gradient clipping | 1.9348 | 1.7828 | Same 155 calls; retain nonfinite checks and clipping behavior. |
| Replay window sampling | 1.2873 | 0.0676 | Same 51 calls; do not infer sampling optimization. |
| Reference FK | 0.5295 | 0.1946 | Same 22 calls and implementation. |
| Tensor nonzero | 0.3673 | 0.4724 | Same 36 calls; dynamic reset compaction deserves timeline inspection. |
| Logging | 0.6510 | 0.1211 | One call each, with diagnostics and synchronization. |
| Environment reset | 0.1039 | 0.0620 | Eight calls each; reset populations were not recorded here. |

The reference run records 1392 delayed-PD compute calls and 4176 delay-buffer compute calls: 48 physics steps × 29 actuators × three delayed target channels. Actuator collection accounts for 1.2169 cumulative host seconds, and delayed-PD compute for 0.9827 nested seconds. This directly confirms the repeated dispatch structure. Grouped actuator functions do not appear in the grouped profile's retained top 100 rows; their absence is not a zero-cost measurement. A complete device trace and repeated matched trials are needed to attribute the total interval difference.

The grouped profile still records 321 reference samples (0.0750 cumulative seconds). `canonical_state` does not appear in its retained top 100. Repeated state cloning and reference gathering are code-review opportunities, but this profile does not establish them as dominant costs.

## Batched FK training attempt

A separate attempt used depth-batched FK at the same full training shape: 2560 environments, world-model batch 2048 × 10, policy batch 4096 and horizon 32. It did not finish the requested five iterations after several minutes and was stopped. A device-monitor sample showed 15966 MiB used, 100% GPU utilization and approximately 79 W power.

This attempt is a failed performance validation at the required training shape. The repeated reference-FK run also showed almost the same device-memory occupancy and subsequently completed slowly. Consequently, the memory/performance issue cannot be attributed specifically to batched FK. These observations are consistent with memory pressure but do not prove host-memory spill, an out-of-memory exception or a particular allocator mechanism. No completed timing or peak-Torch-allocation result is available for it. Batched FK must not become the training default on the strength of a smaller CPU or forward-only benchmark. Actor FK handles 131072 imagined states and participates in backward; verification must include peak memory and the complete actor update.

## Remaining audit targets

Source paths below are relative to `mimic_baseline/source/mimic_baseline/mimic_baseline/tasks/supermimic/`.

| Priority | Target and source | Safe next verification |
| --- | --- | --- |
| High | Actuator grouping: `robots/g1/asset.py`, `envs/evaluation_actuators.py` | Check delay draws and RNG order, reset histories, joint ordering, gains, limits and full-training outputs before enabling outside a benchmark. |
| High | Actor memory and FK: `learning/policy_update.py`, `core/kinematics.py`, `robots/g1/kinematics.py` | Measure full forward/backward peak memory. Require pose, velocity and gradient parity. Recovery already shares FK across reward, hazard and critic inputs. |
| High | Gradient-clipping synchronization: `learning/trainer.py::_clip_gradient_norm` | Device timeline around norm, finite check and backward; preserve error detection and global clipping budget. |
| Medium | Reset compaction: `envs/mdp/commands.py::record_step` | Identify which of the 36 nonzero calls can reuse existing reset IDs without changing ordering, reset populations or RNG. |
| Medium | Replay sampling: `learning/replay.py::_weighted_logical` | Separate cache hits and invalidations. The full candidate CDF is already cached across repeated minibatches; preserve sampled indices, weights and floating-point CDF behavior. |
| Medium | Logging diagnostics: `learning/trainer.py::_log`, `learning/world_model_update.py::world_model_diagnostics` | Label the additional 2048-window, up-to-32-step diagnostic rollout separately. It samples replay and consumes RNG, so changing log cadence is not an exact-parity optimization. |
| Lower until measured | State/reference reuse: `envs/mdp/commands.py::canonical_state`, `envs/manager_based.py`, `learning/collection.py` | Cache only within a verified simulation-state epoch, invalidating on step/reset. Preserve transition ownership. Existing post-state/FK caches must not be duplicated. |
| Lower until measured | Checkpoint I/O and cache release: `learning/checkpoint.py`, `runtime/safety.py` | Measure separately; retain checkpoint cadence and memory settings. The latest-checkpoint alias already avoids duplicate serialization. |

Do not reduce batch sizes, horizon, replay capacity, model updates, collision checks, recovery resets or protection/termination logic as an undocumented performance fix. Many other nonzero sites belong to perception or generated-motion branches and are inactive in this run.

## Bounded evaluation comparison

Two additional full-clean evaluation profiles used grouped actuators, the same checkpoint/source identity above, and a fixed cutoff of 1024 control steps. They were runtime probes, not completed all-clip evaluations. The training-shape metadata retained in their JSON describes the source checkpoint configuration; it must not be interpreted as the number of active evaluation environments.

| Observation | Reference FK | Batched FK |
| --- | ---: | ---: |
| Synchronized interval, 1024 steps | 31.0213 s | 19.3718 s |
| FK calls | 1024 | 1024 |
| FK cumulative host time | 7.8528 s | 1.8508 s |
| Canonical state calls | 6148 | 6148 |
| Canonical state cumulative host time | 1.9560 s | 1.9525 s |
| Reference samples | 7173 | 7173 |
| Reference sample cumulative host time | 1.1112 s | 1.0873 s |
| Peak Torch allocated memory | 0.8496 GiB | 0.8486 GiB |
| Peak Torch reserved memory | 1.5703 GiB | 1.8203 GiB |

This provides direct evidence of repeated FK dispatch cost in evaluation and a shorter bounded interval with batched FK. The same asynchronous-host attribution limits apply; the interval reduction cannot all be assigned to the FK row. It does not establish acceptable full-batch differentiable training memory use.

State extraction is now a measured evaluation target: approximately six canonical-state calls per control step, totaling about 10% of the batched-FK measured interval in cumulative host time. Reference gathering adds another inspectable repeated cost. Cache proposals must distinguish pre-physics, post-physics and post-reset states and preserve observation-noise/RNG behavior and transition ownership. This elevates state/reference reuse for evaluation; the training top-100 profile still does not show it as a dominant training cost.

A final 1024-step profile of the cached evaluation implementation measured **12.7355 s** (peak Torch allocated 0.8486 GiB). Canonical-state calls dropped from **6148 to 3076**, and their cumulative host time from **1.9525 to 1.0278 s**. FK remained essentially unchanged at 1.8508 versus 1.8464 s. The call-count reduction directly confirms the intended ManagerBased optimization; the larger total-wall difference cannot be attributed solely to those host function times because asynchronous scheduling and run-to-run effects still apply.

## Complete-trajectory acceptance

All jobs use checkpoint 6600, protocol version 2, all 40 original LAFAN1 clips, 40 environments per profile, the original first-to-last resampled frames, and a maximum of 13655 control steps. Strict retains native failure termination; full-clean disables termination, truncation, domain randomization, observation/reset noise and action delay. Two concurrent profiles mean two isolated 40-environment simulators, not a split of the 40 clips.

| Complete job | Wall time | Strict completed / return | Full-clean coverage / return |
| --- | ---: | --- | --- |
| Previous fixed grouped-PD job | 437.625 s | 7/40 / 0.342636 | 40/40 / 0.674870 |
| Batched FK + state reuse, sequential | 298.516 s | 7/40 / 0.335632 | 40/40 / 0.696768 |
| Same implementation, parallel | 279.485 s | 6/40 / 0.330480 | 40/40 / 0.701851 |
| Deployed fixed command, parallel repeat | 280.313 s | 7/40 / 0.327054 | 40/40 / 0.697699 |

The new parallel job is about **36% shorter** than the previous fixed-job measurement. Parallelism itself contributes only **6.4%** versus the new sequential job; both processes slow down while sharing the GPU. These are measured examples, not a universal throughput guarantee. Five-second device samples reached 2688 MiB in the observed sequential interval and 4930 MiB across parallel execution; sampling is not a precise allocation-peak measurement.

Generated per-clip reports and exact measured metadata are linked from [the fixed-entry documentation](evaluation-job.md). Both complete reports validated before/after realized runtime settings, expected frame counts, complete clip sets, source hashes and checkpoint identity. No clip was shortened to improve the timing.

Native G1 CUDA checks covered batches 40 and 4096: maximum output discrepancies were 9.54e-7 and 1.43e-6; maximum input-gradient discrepancies were 2.86e-6 and 7.63e-6. The tests also cover MotionLib copying and double-precision loading. GPU snapshot validation checked 384 cached reads over 128 steps in **each** protocol against fresh canonical-state reads with zero absolute/relative tolerance. The complete jobs then exercised the normal cache without validation overhead.

Long Newton rollouts are not bitwise identical, including reference repeats previously recorded in the evaluation benchmark. FK arithmetic is mathematically equivalent within tested floating-point tolerances; the cache comparison is exact. Neither changed strict completion counts nor return differences establish a learning improvement or prove every difference has one particular numerical cause. Full-clean 40/40 is coverage, not tracking success.

The fixed runner now selects grouped actuators, batched FK and parallel scheduling; explicit reference and sequential modes remain available. The deployed immutable release is `47055435fbef35d4ca564509cd218fe447abcc14`. Formal training remains stopped. Production training retains reference FK and its original actuator default; the profiling CLI exposes the measured alternatives only for isolated, bounded audits.
