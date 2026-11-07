"""Restore the original Trainer configuration without silently changing a run."""


def restore_run_config(checkpoint, overrides, *, small_model=None):
    if checkpoint.get("format") != "supermimic-g1-v1":
        raise ValueError("Expected an original ManagerBased G1 checkpoint")
    config = dict(checkpoint["config"])
    config.setdefault("self_collisions", checkpoint.get("provenance", {}).get("backend", "ovphysx") != "newton")
    if config.get("robot_name") != "g1" or config.get("perception", False):
        raise ValueError("Expected proprioceptive G1 configuration")
    saved_small = bool(checkpoint.get("provenance", {}).get("small_model", False))
    if small_model is not None and small_model != saved_small:
        raise ValueError("small_model differs from checkpoint architecture")
    relocatable = {"iterations", "log_dir", "device", "urdf", "robot_usd", "motion_file", "save_every", "log_every"}
    for name, value in overrides.items():
        if name not in relocatable and value != config.get(name):
            raise ValueError(f"{name} differs from checkpoint; resume preserves training settings")
        config[name] = value
    return config, saved_small


def resolve_backend(requested, checkpoint):
    saved = checkpoint.get("provenance", {}).get("backend", "ovphysx") if checkpoint else None
    backend = requested or saved or "ovphysx"
    if backend not in ("newton", "ovphysx"):
        raise ValueError(f"Unknown backend: {backend}")
    if saved is not None and backend != saved:
        raise ValueError("backend differs from checkpoint; resume preserves physics")
    return backend
