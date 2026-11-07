# Recorded Newton LAFAN1 result, 2026-11-05

- Isaac Lab v3.0.0, commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`.
- RTX 5070 Ti, Windows, Newton MJWarp, 4,096 environments.
- Dataset: all 40 G1 LAFAN1 clips, revision `ce1572906efe6157840e8474d5a0d7aa87481e74`.
- Profile: `supermimic_g1/train_newton_lafan1.cmd`.

| Iteration | Completed clips | Normalized return |
| ---: | ---: | ---: |
| 1,000 | 25/40 | 0.3809 |
| 4,000 | 32/40 | 0.6430 |
| 11,000 | 38/40 | 0.7992 |
| 13,000 | 40/40 | 0.826475 |

At 13,000, frame-weighted return was 0.829880, mean clip fraction 1.0, and the failed-clip list was empty.

## Evidence and limits
The 13,000 result was read from the Windows run's best_eval.json before shutdown. Its checkpoint is ckpt_013000.pt. The latest saved checkpoint observed during shutdown was ckpt_014000.pt; its evaluation was not confirmed.

Training and evaluation were stopped at the user's request. Those checkpoints remain on the original Windows machine. Their SHA-256 hashes have not been collected. The code was not under Git during training, so a training-time commit cannot be supplied retroactively.

These are training-set tracking results, not held-out generalization measurements. This Git publication organizes source and records prior observations; it does not rerun GPU training.
