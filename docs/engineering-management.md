# Engineering workflow

## Initial publication
- [x] Keep the current G1-only training path as the public project.
- [x] Pin Isaac Lab as an upstream submodule.
- [x] Exclude historical company material, credentials, datasets and model artifacts.
- Validate the publication inventory and Python compilation before each publication.
- Commit, push and compare remote/local commit IDs before reporting synchronization.

## Daily work
Use a task branch, implement a reviewable change, run relevant checks, review the staged diff, commit and immediately push. Repeat for each validated change. Integrate a completed task through a pull request.

Publication scope is the allowlist in .gitignore. Review newly included files when expanding it. Historical local directories remain outside Git.

## Experiments
Record code/runtime/data revisions, training arguments, checkpoint location and SHA-256, metrics and validation scope. Keep large artifacts outside Git. Redact machine-specific paths and private infrastructure.

CPU CI compiles source and checks publication hygiene. GPU execution requires an explicit check on a suitable NVIDIA machine. Passing CPU CI does not establish training quality. Agent instructions are in AGENTS.md.
