# ManagerBased G1 architecture rebuild

Goal: prepare a safe standalone all-40 LAFAN1 evaluation command and deeply reorganize the original ManagerBased implementation without changing learning, physics, reset/termination semantics or checkpoint schema. Multi-backend expansion is excluded.

Keep the Isaac Lab extension package root and Python namespace for compatibility. Within `tasks/supermimic`, canonical implementations will live under `core/`, `data/`, `learning/`, `envs/`, `robots/g1/`, `evaluation/`, `runtime/` and `cli/`. Existing leaf-module imports and script paths will remain thin compatibility adapters. Add clear top-level script entry points and an architecture guide; do not move or publish historical private assets. The standalone historical trainer remains explicitly separate.

1. Add standalone evaluation with checkpoint/data preflight, new-output protection, exactly-40 clip validation, checkpoint digest/protocol metadata and aggregate/per-clip artifacts. Test failure gates and reporting; commit/push.
2. Move numerical, motion/data, robot and learning modules into canonical packages. Replace old modules with identity-preserving aliases, repair imports mechanically and freeze baseline function AST contracts. CPU parity/serialization tests; commit/push.
3. Split original Trainer into configuration, instrumentation, collection/orchestration, world-model update, policy/critic update and checkpoint responsibilities without rewriting method bodies. Split ManagerBased configuration/managers/environment/adapter and evaluation rollout/reporting. Verify compatibility contracts and tests; commit/push.
4. Separate CLI configuration/composition from shell entry points; explicit Gym registration must not recursively import simulator modules on CPU. Document dependency boundaries, canonical paths, legacy compatibility and commands; commit/push.
5. Export a separate Windows validation release only after the ongoing formal run and its automatic evaluation have finished. Compare old/new evaluator outputs on the same checkpoint, run original full-network short train/resume, and validate Newton environment behavior. Independent review, CI and PR integration.

The deployed formal run at source `1bf1a4e` is immutable and must not be overwritten. Its 3000-iteration and all-40 evaluation results remain distinct from reconstruction smoke tests. Every independently validated change is committed and pushed immediately.
