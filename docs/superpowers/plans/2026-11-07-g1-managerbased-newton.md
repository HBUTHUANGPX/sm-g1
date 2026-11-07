# ManagerBased G1 Newton and 3000-iteration training

Goal: retain the original managers and Trainer, select Newton with MJWarp, train full networks for 3000 iterations and evaluate all 40 LAFAN1 clips.

Architecture: add a backend selector to the existing G1 config/CLI, preserve 50 Hz control and 200 Hz physics, and implement only the Newton material bridge in the existing randomization helper. Keep original randomization ranges and learning equations. Do not import the historical standalone trainer.

User authorized execution and immediate commit/push of validated units. Initial capacity candidate is 512 environments; select final size from GPU memory and throughput measurements. Checkpoint every 100 iterations, record code/data/runtime identity, evaluate the final checkpoint on all original unpadded clips. No convergence promise.

- [ ] Backend selection and resume compatibility tests, implementation and GPU manager checks.
- [ ] Newton material/COM reset checks, preserving nonselected environments and nominal COM.
- [ ] Full-model capacity and throughput trial, select environment and batch sizes with evidence.
- [ ] Formal 3000 iterations, finite-state/optimizer monitoring and checkpoint verification.
- [ ] Evaluate all 40 clips, publish exact results, review and integrate through PR.

Failure focus: mismatched checkpoint backend; material writes affecting other environments; wrong physics/control time ratio; exhausted GPU memory; evaluated clip endpoints accidentally including padding.
