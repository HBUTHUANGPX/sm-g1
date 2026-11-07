"""Independent SuperMimic tracking for RX-P2.

The training implementation does not import the historical RGMT/ERP adapters.
See README.md for paper settings and explicit robot/backend adaptations.

ManagerBasedRLEnv entry points
-------------------------------
``SuperMimicRLEnv``  — flat-ground, proprioception-only (no perception).
``SuperMimicErpRLEnv`` — fixed mixed terrain with ERP depth camera.

Both use the original model-based SuperMimic Trainer, not a PPO runner.
The original TrackingSim remains available for baseline checkpoint evaluation.
"""

import gymnasium as gym

_ENTRY = "mimic_baseline.tasks.supermimic.env_managed"
_CFG = "mimic_baseline.tasks.supermimic.env_managed"

gym.register(
    id="SuperMimic-Flat-RXP2-v0",
    entry_point=f"{_ENTRY}:SuperMimicRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG}:SuperMimicEnvCfg",
    },
)

gym.register(
    id="SuperMimic-ERP-RXP2-v0",
    entry_point=f"{_ENTRY}:SuperMimicErpRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_CFG}:SuperMimicErpEnvCfg"},
)

# G1 uses the same ManagerBased environment and model-based Trainer.
gym.register(
    id="SuperMimic-Flat-G1-v0",
    entry_point=f"{_ENTRY}:SuperMimicRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "mimic_baseline.tasks.supermimic.g1_env:G1SuperMimicEnvCfg"},
)


gym.register(
    id="SuperMimic-Recovery-G1-v0",
    entry_point=f"{_ENTRY}:SuperMimicRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "mimic_baseline.tasks.supermimic.g1_env:G1RecoveryEnvCfg"},
)
