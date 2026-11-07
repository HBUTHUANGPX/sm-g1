# Reusable G1 evaluation job

Run from the pinned environment with the Python interpreter that has Isaac Lab and the selected physics backend installed. `--help` requires no simulator. Commit and push the evaluated source first, or use an exported release whose manifest hashes match its files. Modified or unmanifested release Python files are rejected.

Create an **untracked, local** JSON configuration outside the repository (paths below are placeholders):

```json
{
  "data_root": "/local/g1-data",
  "checkpoint_root": "/local/recovery-run",
  "output_root": "/local/evaluation-jobs"
}
```

Relative config paths resolve against the config file directory. The runner uses the active Python interpreter; there is no separate Python setting. The config, data, checkpoints, logs and full outputs must remain outside publication scope.

```sh
python scripts/run_g1_evaluation.py --config /local/evaluation.json
```

This chooses the highest numeric `model_N.pt` directly in `checkpoint_root` or its `train` subdirectory. Ties are rejected. To select a particular saved checkpoint:

```sh
python scripts/run_g1_evaluation.py --config /local/evaluation.json --checkpoint /local/recovery-run/train/model_6600.pt
```

The default is both protocols with grouped execution. `--profile strict` or `--profile full_clean` selects one protocol. `--execution reference` selects the original per-joint actuator implementation. Checkpoint ABI, assets and all 40 source clips are validated by the evaluation CLI before simulation.

Every invocation creates a unique output directory; no cache or old output is reused. `status.json` records running, failed or complete, elapsed seconds, checkpoint hash and source identity. Each profile retains its simulator log, manifest, per-clip results and summary under `evaluation/`. On successful validated evaluation, the runner writes combined `summary.json` and `report.md` and prints the report. The report includes all 40 clip rows for each requested protocol. Failed jobs retain their directory and error status; rerunning creates a new directory.

Strict completion uses native termination. Full-clean coverage measures reaching the original clip endpoint with termination and actuator delay disabled; it is **not tracking success**. The report revalidates clip coverage and before/after observed runtime checks rather than assuming requested flags were realized. Source and checkpoint identity are checked again after evaluation. Outputs are local artifacts: review and sanitize any compact summary before publication.

During rollout, `progress.json` reports vector steps, active clips, elapsed time and an upper-bound ETA every 256 steps. The parent command forwards these updates to the terminal; per-profile `elapsed_seconds` includes preflight/startup/rollout. The grouped mode batches PD and delay storage without sharing delays between joints, and keeps the original per-joint random draw order. Training actuators are unchanged.

The validated fixed-job scope is the current G1 delayed-PD recovery checkpoint. Grouped execution rejects other actuator types instead of mislabelling a fallback. Source/model hashes do not fingerprint every installed runtime package or external asset; retain the pinned Isaac Lab/data setup. Native actuator parity checks are exact, but complete Newton rollouts are not promised bitwise reproducible; same-checkpoint result variation is not a learning improvement.

## Measured acceptance

On checkpoint6600 / RTX5070Ti / Windows Newton, the complete two-profile job took 437.625 seconds (7.3 minutes), compared with 1114.982 seconds (18.6 minutes) for the prior run: about 2.55x faster. Native CUDA checks compared 92,800 joint transitions exactly, including gains/limits, independent delays, partial resets and RNG state. See [benchmark evidence](results/2026-11-08-g1-evaluation-performance.json) and [the unedited generated report](results/2026-11-08-g1-fast-evaluation.md). This is a measured example, not a cross-machine guarantee.

The operational entry point pins fully evaluated release `1c50bb87c19f2ffcffa1d737ef1a962670f345e9`; later unsupported-mode guards do not change the validated delayed-PD execution path. Upgrade a deployed release deliberately rather than modifying its verified files in place.
