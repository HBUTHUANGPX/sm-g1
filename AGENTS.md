# Project working agreement

## Scope
- Active project: `supermimic_g1/`, Unitree G1, proprioceptive SuperMimic, LAFAN1 only.
- PUBLIC repository: https://github.com/HBUTHUANGPX/sm-g1.
- Isaac Lab is pinned to v3.0.0 as a submodule. Keep project adaptations outside it.
- Historical RX-P2 code and company documents in this workspace are outside the publication allowlist.
- Keep Windows training stopped and SSH disconnected until the user requests remote work again.

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
