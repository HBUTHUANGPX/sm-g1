# ManagerBased G1 source architecture

The active implementation retains the Isaac Lab extension namespace
`mimic_baseline.tasks.supermimic`. All paths below are relative to
`mimic_baseline/source/mimic_baseline/mimic_baseline/tasks/supermimic/`.
This reorganization preserves the original ManagerBased lifecycle, learning
methods, network sizes, checkpoint keys and recovery semantics. It introduces
no new simulator abstraction or learning backend.

| Directory | Responsibility |
|---|---|
| `core/` | Analytic MDP, rotations, action bounds, recovery math and base kinematics |
| `data/` | LAFAN1 conversion, motion loading, reference providers and sampling |
| `robots/g1/`, `robots/rxp2/` | Robot metadata, kinematics and asset configuration |
| `envs/configs/` | Base and G1 ManagerBased configurations |
| `envs/mdp/` | Native action, command, event, observation, reward and termination terms |
| `envs/managers.py` | Transition capture and four-group probabilistic termination |
| `envs/manager_based.py` | Original environment lifecycle and reset behavior |
| `envs/adapter.py` | Original transition API exposed to the model-based Trainer |
| `envs/actuators.py`, `randomization.py` | Delay buffers and physical randomization |
| `envs/simulation.py` | Shared simulation construction defaults |
| `learning/` | Networks, replay, configuration, Trainer composition and update methods |
| `evaluation/` | Rollout execution, tracking metrics and report serialization |
| `runtime/` | Configuration restoration, runtime checks and release paths |
| `cli/` | Argument parsing and application composition |
| `validation/` | Real simulator checks for the G1 manager integration |

`Trainer` owns model, optimizer, replay and environment state. Its collection,
world-model update, policy/critic update and checkpoint methods are grouped in
single-purpose mixins; these are not alternative trainers. The method bodies are
retained from the original implementation. Configuration and logging can be
imported on CPU without importing Isaac Lab or starting simulation.

Dependency direction is CLI → runtime composition → environment/learning/data →
core math. Simulator APIs belong to robot assets and environment modules. Keep
`core`, data math and model construction importable without Isaac Lab. The
package initializer only registers explicit Gym entry points. Optional perception
implementation remains separate and is outside the supported G1 entry point.
`envs/legacy.py` and the root `supermimic_g1/` are historical implementations;
ManagerBased G1 does not delegate its physics loop to them.

## Commands and compatibility

Use root entry points from the repository or exported release:

```bash
python scripts/train_g1.py --help
python scripts/evaluate_g1.py --help
python scripts/run_g1_experiment.py --help
python scripts/export_g1_release.py --help
```

The previous `mimic_baseline/scripts/supermimic/` command paths remain wrappers.
Old leaf imports such as `training`, `env_managed`, `g1_env`, `motion` and
`mdp_legacy` resolve to the same canonical implementations. New code should import
canonical modules. `.github/g1-module-map.json` records these aliases. Evaluation
is now a package that reexports its established public functions.

Checkpoint format remains `supermimic-g1-v1`. No state-dict key migration is
required. Configuration restoration still rejects incompatible architecture and
physics overrides. Resume refills replay using the original behavior; it is not a
bitwise continuation of saved simulator state.

## All-40 evaluation

```bash
python scripts/evaluate_g1.py --checkpoint runs/train/model_3000.pt --data-root supermimic_g1/data --output runs/eval-3000 --dry-run
python scripts/evaluate_g1.py --checkpoint runs/train/model_3000.pt --data-root supermimic_g1/data --output runs/eval-3000
```

Use a **new output directory**. Preflight checks all 40 finite 36-column source
CSVs, nonzero quaternions, required asset files, supported saved configuration,
finite policy tensors and strict CPU policy loading. It does not validate USD
references or prove GPU runtime compatibility; those require actual simulation.
Dry-run creates no output directory and launches no simulator.

The command infers backend/model configuration from the checkpoint. It evaluates
from each clip's first frame to its original unpadded endpoint. Observation/reset
noise and physical randomization are disabled; actuator delay follows the saved
configuration and seeded reset behavior. Tracking failure termination is
deterministic, with no training recovery protection period. This protocol is
unchanged by the source reorganization.

Outputs: `evaluation_manifest.json`, `evaluation.log`, `perclip.json`,
`perclip.csv`, `by_category.csv`, and `summary.json` containing checkpoint SHA256,
iteration, all-40 coverage, completion count and failed clip names. A failed
execution/validation produces `failure.json` and no success summary. Completion
of the command does not imply that the policy completed the motions.

## Validation and release discipline

CPU suite:

```bash
uv run --with torch --with pytest --with numpy --with pyyaml --with gymnasium python -m pytest mimic_baseline/tests/g1 -q
python .github/check_repository.py
```

The behavior contract fingerprints original function bodies independently of
import relocation and formatting. It catches accidental numerical/control-flow
changes, but does not replace simulator validation or cover every class-level
configuration. CLI/import checks use the real package initializer.

Exports consume only the committed publication allowlist and record file SHA256s.
Keep every running experiment's exported source immutable until its automatic
evaluation finishes. Deploy architecture validation to a separate directory;
compare evaluation on the same checkpoint and verify full-network train/resume
before treating the reorganization as runtime-validated. Never copy generated
assets, datasets, checkpoints or machine connection details into Git.

## Runtime acceptance (2026-11-07)

Validated exported source `ce85602d96c237aaa9e21c4e8551a790833f8197` on the
existing Windows Newton runtime; all 139 exported file hashes matched. Later
commits record results and documentation only. See [machine-readable evidence](results/2026-11-07-g1-architecture.json).

- Mac CPU: 45 passed. Windows CPU: 44 passed, one Git-only export test skipped
  because Git is absent there; export was tested on Mac/CI. GPU regression: 5 passed.
- Original function/configuration contracts: 395.
- Newton recovery environment: 64 environments, 205 finite steps; delay ticks
  0–4, upside-down resets, collision clearance, self-contact, probabilistic groups,
  protection boundary, evaluation termination and partial reset checks passed.
- Full original networks, 32-step BPTT: train to iteration 2, resume to 4; a
  pre-refactor checkpoint resumed from 12 to 14. All three optimizer counters
  and finite parameters were verified. These are execution tests, not convergence.
- Same model_3000 checkpoint and all 40 clips evaluated with both source versions.

| Evaluation | Completed clips | Mean normalized return |
|---|---:|---:|
| Original automatic run | 0/40 | 0.06922115 |
| Original unchanged-source repeat | 0/40 | 0.06601041 |
| Reorganized source | 0/40 | 0.07272094 |

**Numerical reproducibility limit:** the unchanged old version itself changed
termination steps on six clips, by up to 3036 steps. Old versus reorganized
source changed four clips, by up to 3169 steps. Thus these results do not establish
bitwise trajectory parity or statistical equivalence; three evaluations are not
a performance comparison. No success outcome changed, and no improvement is
claimed. Numerical/manager method bodies and configuration classes remain
protected by the original-source contracts. This task did not alter physics
settings to force repeatability. No physical robot was tested.
