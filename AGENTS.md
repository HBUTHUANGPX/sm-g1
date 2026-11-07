# Project working agreement

## Scope
- Active migration: introduce Unitree G1 into the ORIGINAL `mimic_baseline/tasks/supermimic/envs/manager_based.py` (legacy `env_managed.py` import remains supported) ManagerBased path, reusing its managers and Trainer. Proprioception and LAFAN1 only.
- `supermimic_g1/` is the previous standalone implementation, retained as a historical comparison. Do not substitute it for the original ManagerBased path. Multi-backend work is deferred.
- PUBLIC repository: https://github.com/HBUTHUANGPX/sm-g1.
- Isaac Lab is pinned to v3.0.0 as a submodule. Keep project adaptations outside it.
- Historical RX-P2 code and company documents in this workspace are outside the publication allowlist.
- On 2026-11-07 the user stopped the initial Newton run (last saved iteration 2300) and authorized G1 recovery migration: action delay, Newton self-collision, large-pose resets, four-group probabilistic termination/protection and dimension fixes. TN is explicitly excluded. The subsequent user request authorizes a NEW 3000-iteration Newton recovery run, the largest practical validated environment count, and automatic evaluation of all 40 LAFAN1 clips. Preserve the stopped flat experiment; do not resume it as recovery.

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
