# Newton reset-history short training validation — 2026-11-12

**The bounded run completed successfully: checkpoint 6600 → 6610, 10 learning iterations. Formal training was not resumed.** Only the reset-history fix and opt-in diagnostic observer are new. The run uses the original reference actuator/FK execution and default native allocator, in an isolated benchmark directory outside production checkpoint discovery.

## Preserved configuration and provenance

- Code: `3d0ad90155521ffa34b12eda47c530be7e3f7b74`, incorporating reset-history fix `85e9fd08170aa39e4acfff898629d0aa130f6081`.
- Verified release manifest SHA256: `a0990697368afc881997624900653bb86bb1c83f8d29aa94fa246ec0c1c828f3`.
- Starting checkpoint SHA256: `c4d4c8188f91c4683464e64bf7ccb5c4c99047386f906950e5e7e4253705cf51`. It was rehashed after the run and remained unchanged.
- Separate ending checkpoint SHA256: `3b4f4328e0356014b8e46c6f53ab7fea22990bae56cc778489d22ccd156dec4a`.
- Newton, 2560 environments, recovery enabled, delayed PD with max delay 4; policy batch4096/horizon32; WM batch2048×10, window8; replay1000, refill64, log_every10. Exact checkpoint configuration comparison changed **only `iterations` and `log_dir`**.
- Runtime: RTX5070Ti, Python3.12.16, Torch2.11.0+cu130, Newton1.6.1, Warp1.17.0, MuJoCo3.12.0. Isaac Lab source remains pinned v3.0.0; installed package metadata reports28.0.0, as in previous audits.

## Observation, loss and gradient checks

The observer does not alter losses, gradients, random draws, update counts or optimizer settings. Hooks are restored even on failure. It introduces CPU/GPU synchronization; timing from this run is diagnostic, not a speed claim.

- Ten imagined-history checks: shape `[4096,10,93]`, **40,960 rows**; all finite.
- **39,259 adjacent duplicate-frame pairs** were checked for exact equality; all passed. This checks actual replay-sampled histories during CUDA training, including noisy padding.
- Eleven real-collection boundary history checks, including the refill: all finite. These observe the last history at each collection call, not every intermediate physics step. Real/imagined correlation semantics are separately covered by the existing CPU regression suite.
- All **310** pre-clipping gradient norm checks were finite: 100 WM, 10 actor, 200 critic. Norm clipping remains1 and actor element clipping remains.1.

| Metric | Observed minimum | Observed maximum | Checks |
|---|---:|---:|---:|
| WM loss | 63.3750 | 103.5308 | 100 |
| Actor loss | -5.74761 | -5.56275 | 10 |
| Critic loss | 1.13539 | 2.04317 | 10 |
| WM gradient norm, before clipping | 111.7926 | 227.4992 | 100 |
| Actor gradient norm, before clipping | 2.77135 | 4.72459 | 10 |
| Critic gradient norm, before clipping | 7.88987 | 96.66457 | 200 |

No recorded loss was NaN/Inf. Critic loss is the original final-minibatch metric returned by each policy/critic update, not a separate loss record for every minibatch; all200 critic gradient updates were nevertheless checked. Gradients frequently exceed the clipping budget before clipping, so the finite result should not be described as “no clipping needed.” This short run does not establish convergence or improved tracking.

## Memory and completed artifacts

- Whole-audit peak PyTorch allocated: **11.5597 GiB**; reserved: **14.1152 GiB**.
- CUDA allocator OOM events: **0**; allocation retries: **0**, both before and after the measured window.
- CUDA reported **0 free device bytes** at the sampled iteration boundaries. There was no allocation failure in this run, but memory headroom remains tight; this is not evidence of long-run memory stability.
- Five warmed diagnostic iterations took11.1439s (includes final normal logging/checkpoint I/O); refill15.1415s. Do not compare these directly to throughput-only windows.
- Ending WM/actor/critic tensors **and all three optimizer states** are finite. Model shapes unchanged. Exact increments:10 iterations,310 optimizer steps,227840 real transitions including refill,204800 WM windows.
- Final process inspection: no Python processes remain; GPU returned to idle at446MiB used. No persistent training or monitoring job was installed.

## Reusable invocation

```sh
python scripts/profile_g1_runtime.py \
  --checkpoint <original-checkpoint> --data-root <data-root> --output <fresh-benchmark-directory> \
  --target training --iterations 10 --warmup-iterations 5 --no-cprofile \
  --actuators reference --kinematics reference --verify-training-health
```

Local CPU suite: **183 passed, 4 runtime-dependent skips**, including diagnostic hook restoration and rejection of corrupt padding/nonfinite observations, losses and gradients. The GPU run above supplies actual Newton acceptance. Other differences from the official SuperMimic implementation remain untouched.

[Machine-readable evidence](2026-11-12-newton-reset-history-short-check.json). Checkpoints, raw logs, connection details and machine paths are not published.
