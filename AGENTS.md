# Project working agreement

## Scope
- Active migration: introduce Unitree G1 into the ORIGINAL `mimic_baseline/tasks/supermimic/envs/manager_based.py` (legacy `env_managed.py` import remains supported) ManagerBased path, reusing its managers and Trainer. Proprioception and LAFAN1 only.
- `supermimic_g1/` was the previous standalone implementation, removed from the active tree and retained in Git history at `4727620`. Reusable asset tools live in `tools/assets/`. Do not substitute it for the original ManagerBased path. Multi-backend work is deferred.
- PUBLIC repository: https://github.com/HBUTHUANGPX/sm-g1.
- Isaac Lab is pinned to v3.0.0 as a submodule. Keep project adaptations outside it.
- Historical RX-P2 code and company documents in this workspace are outside the publication allowlist.
- On 2026-11-07 the user stopped the initial Newton run (last saved iteration 2300) and authorized G1 recovery migration: action delay, Newton self-collision, large-pose resets, four-group probabilistic termination/protection and dimension fixes. TN is explicitly excluded. The subsequent user request authorizes a NEW 3000-iteration Newton recovery run, the largest practical validated environment count, and automatic evaluation of all 40 LAFAN1 clips. Preserve the stopped flat experiment; do not resume it as recovery.

- The recovery continuation toward 20000 was stopped at the user's request (last saved checkpoint 6600). The current request authorizes bounded training-performance comparisons only, preserving the 2560-environment training configuration. Do not resume formal training. Keep benchmark checkpoints in separate directories outside the production checkpoint-selection root.

## Required commit and push cycle
The user explicitly authorizes ongoing commits and pushes for this project.
1. Check Git status and preserve unrelated user work.
2. Start a short task branch from current main.
3. Implement one independently reviewable change and run relevant checks.
4. Review staged changes for credentials, private/company material, large artifacts and unintended edits.
5. Immediately commit the validated change and push the task branch. Do not accumulate validated changes until the end.
6. Repeat this cycle for each independently validated change.
7. Integrate completed work through a pull request after task-level checks pass. The approved workflow authorizes integration of completed in-scope work.
8. Report commit, branch, push result and validation evidence.

If validation or pushing fails, report it and retain local work. Verify local and remote commit IDs before claiming synchronization. Do not force-push or rewrite shared history. Do not auto-commit partially written files with a filesystem watcher.

## Reproducibility and publication
- Track code, configurations, runtime/data pins, preparation scripts and compact evaluation summaries.
- Exclude data, full logs, checkpoints, virtual environments, caches and credentials.
- Record code/runtime/data revisions, training arguments, checkpoint location/hash and evaluation scope for experiments.
- Download only LAFAN1 via the pinned script. Do not add SOMA.
- Retain licenses and attribution.
- Never publish passwords, tokens, private keys, machine-specific connection details or company infrastructure.

## Fixed evaluation workflow
- Use `scripts/run_g1_evaluation.py --config <local-config>` for the supported G1 delayed-PD recovery checkpoints; `--checkpoint` selects a specific saved model, otherwise the configured root supplies the latest numeric checkpoint.
- Default: grouped execution, both strict and full_clean, all 40 clips at original lengths. `--execution reference` retains the original per-joint evaluation implementation for comparison. Training execution is selected separately; evaluation defaults are unchanged by the training-performance work.
- Reuse the deployed, verified entry point and existing private local configuration. Do not invent per-request rollout/PowerShell/report scripts or manually assemble result tables when the fixed entry point suffices.
- Require job `status.json` complete and validated `summary.json` / `report.md`; display those results. Full-clean coverage is not tracking success. Log failure artifacts honestly.
- Machine paths, connections, data, checkpoints and full logs stay outside publication. Local `*.local.json` configuration must never be committed.

## Training performance workflow
- Original training configuration and Trainer remain authoritative. Execution-only switches are `--training-execution grouped` with `--kinematics-execution lean`; use `bounded --fk-chunk-size 16384` when trading recomputation for lower FK memory. Explicit `reference` switches remain available; CLI defaults remain reference pending stable end-to-end acceptance.
- Use `scripts/profile_g1_runtime.py` for bounded comparisons with the same checkpoint and original shapes. Use 10 iterations, 5 warmup iterations and `--no-cprofile` for throughput comparisons; this window includes final normal logging/checkpoint I/O.
- Do not promote isolated FK speedups or one short training trial into a stable whole-training improvement. Record allocator settings, allocated versus reserved memory, runtime versions and numerical checks. See `docs/results/2026-11-09-g1-training-runtime.md`.
