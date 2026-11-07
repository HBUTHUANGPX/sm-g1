# G1 evaluation protocols (version 2)

Use the same ManagerBased G1 checkpoint for both protocols:

```sh
python scripts/evaluate_g1.py --checkpoint runs/train/model_20000.pt --data-root data/g1 --output runs/eval-20000-v2 --profile both --dry-run
python scripts/evaluate_g1.py --checkpoint runs/train/model_20000.pt --data-root data/g1 --output runs/eval-20000-v2 --profile both
```

An existing external asset directory is accepted through `--data-root`; relocation is not required. Outputs must be new directories. `--dry-run` validates the checkpoint policy and assets on CPU without starting simulation or creating the output.

| Setting | strict | full_clean |
|---|---|---|
| Original ManagerBased environment/managers | Yes | Evaluation-only subclass |
| Four recovery tracking termination groups | Enabled, deterministic failures | Disabled |
| Motion-end and timeout terms | Enabled; timeout beyond longest clip | Disabled |
| Domain randomization / reset noise / observation noise | Disabled | Disabled |
| Action delay | Checkpoint bounds, seeded sampling | Zero for every realized actuator buffer |
| Per-clip rollout | First frame until failure or original last frame | First frame through original last frame |
| Report | Tracking completion, failures, return and errors | Coverage, return and errors; tracking success is not inferred |

The clean profile uses an empty native termination manager, disabling both termination and truncation. It does not merely ignore done flags. Runtime checks verify nominal context, disabled noise and recovery reset roles, realized actuator lags, empty termination terms, finite active state/actions/rewards, and a continuous episode counter without autoresets. A failed assertion aborts evaluation and produces a failure record.

Every source clip must be present exactly once (40 total). Clean results additionally require the exact number of transitions derived from original CSV length after resampling, before motion padding. Shorter completed rows stop accumulating metrics at their own endpoint while other vectorized rows continue.

Version 2 fixes the earlier evaluator's padded-end boundary: short clips could previously continue into appended frames. Historical results retain their old protocol and must not be compared as if the measurement definition were identical. Training and checkpoint parameter shapes are unchanged. The two permitted evaluation-related function changes are explicitly listed in `.github/g1-behavior-contract.json`; the other 393 original contracts remain checked.

Each profile writes `evaluation_manifest.json`, `runtime_profile.json`, `perclip.json`, `summary.json`, and its execution log. The checkpoint hash, iteration, protocol version, expected clips and effective delay are recorded. A clean trajectory reaching its end is **coverage, not tracking success**: a fallen robot may still run to the endpoint, so inspect tracking errors and return alongside the strict result.

CPU checks do not establish Newton GPU compatibility. For the ongoing 20,000-iteration run, the user selected deferred GPU acceptance: a separate immutable evaluation release waits for the original run's complete status (including its original automatic evaluation) before starting the two new profiles. Until that job succeeds, GPU acceptance remains pending.
