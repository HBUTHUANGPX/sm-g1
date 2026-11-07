# G1 recovery Newton formal experiment

Goal: train the original ManagerBased G1 recovery configuration from scratch for 3000 iterations, then evaluate every original LAFAN1 clip.

Keep the validated recovery/actuation/physics settings, full networks and original optimizer batch sizes. Use bounded capacity trials to select the largest practical environment count with GPU headroom; do not infer capacity from the earlier flat run. Keep the stopped flat checkpoint separate.

- [ ] Verify remote identity, no competing training, deployed source and available GPU memory.
- [ ] Run full-configuration capacity trials starting at 1024 environments; record memory, finite parameters and throughput, then adjust upward/downward.
- [ ] Pin source/runtime/data and selected arguments; validate a portable train-then-evaluate launcher with a bounded run; commit and push.
- [ ] Launch fresh 3000-iteration training, saving every 100 iterations; only evaluate model_3000.pt after successful completion.
- [ ] Verify all 40 evaluation records, save compact results/checkpoint hash, commit/push and integrate via PR.
