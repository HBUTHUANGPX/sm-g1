"""G1 entry point using the ORIGINAL ManagerBased environment and Trainer.

This file only prepares inputs, composes existing components and invokes them.
It contains no substitute simulator loop or learning algorithm.
"""

from pathlib import Path
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kinematics-execution", choices=("reference", "batched", "bounded", "lean"), default="reference")
    p.add_argument("--training-execution", choices=("reference", "grouped"), default="reference")
    p.add_argument("--fk-chunk-size", type=int, default=4096)
    p.add_argument(
        "--evaluation-profile", choices=["strict", "full_clean"], default="strict"
    )
    p.add_argument(
        "--evaluation-execution", choices=["reference", "grouped"], default="reference"
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["prepare", "check", "train", "evaluate"], default="check"
    )
    p.add_argument(
        "--recovery",
        dest="recovery_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--actuator-type", choices=["implicit", "delayed_pd"], default=None)
    p.add_argument("--max-delay", type=int, default=None)
    p.add_argument("--backend", choices=["ovphysx", "newton"], default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--small-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="64-wide WM/policy for integration checks, not quality claims",
    )
    p.add_argument("--clip-limit", type=int, default=0)
    for name in (
        "buffer_steps",
        "wm_batch",
        "wm_batches",
        "policy_batch",
        "horizon",
        "warmup_steps",
        "critic_epochs",
        "critic_minibatches",
        "save_every",
        "log_every",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    args = p.parse_args()
    if args.kinematics_execution == "batched" and args.mode != "evaluate":
        p.error("Batched kinematics is supported only in evaluate mode")
    if args.mode != "evaluate" and args.evaluation_profile != "strict":
        p.error("--evaluation-profile is only valid in evaluate mode")
    if args.kinematics_execution in ("bounded", "lean") and args.mode not in ("train", "check"):
        p.error("bounded/lean kinematics is only valid for training/check mode")
    if args.training_execution != "reference" and args.mode not in ("train", "check"):
        p.error("--training-execution is only valid for training/check mode")
    if args.fk_chunk_size <= 0:
        p.error("--fk-chunk-size must be positive")
    import torch
    from mimic_baseline.tasks.supermimic.data.lafan1 import convert_clip, ensure_clip

    data = args.data_root.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted((data / "lafan1_g1").glob("*.csv"))
    if args.clip_limit:
        csvs = csvs[: args.clip_limit]
    if not csvs:
        raise ValueError("No LAFAN1 G1 CSV files")
    output = data / "lafan1_manager_npz"
    paths = []
    for source in csvs:
        target = output / (source.stem + ".npz")
        if args.mode == "prepare":
            convert_clip(source, target)
        else:
            ensure_clip(source, target)
        paths.append(target)
    if args.mode == "prepare":
        print(json.dumps({"converted_clips": len(paths)}))
        return
    from isaaclab.app import launch_simulation
    from mimic_baseline.tasks.supermimic.robots.g1.kinematics import (
        G1Kinematics, G1BatchedKinematics, G1TrainingKinematics, G1LeanKinematics,
    )
    from mimic_baseline.tasks.supermimic.envs.configs.g1 import G1SuperMimicEnvCfg
    from mimic_baseline.tasks.supermimic.envs.manager_based import (
        SuperMimicRLEnv,
        SuperMimicTrainingEnv,
    )
    from mimic_baseline.tasks.supermimic.data.motion import MotionLib
    from mimic_baseline.tasks.supermimic.core.mdp import (
        MDP,
        unflatten_state,
        normalize_xi,
        INV_DIM,
        CRITIC_EXTRA,
    )
    from mimic_baseline.tasks.supermimic.learning.models import (
        WorldModel,
        Policy,
        Critic,
    )
    from mimic_baseline.tasks.supermimic.learning.replay import ReplayBuffer
    from mimic_baseline.tasks.supermimic.learning.trainer import (
        Trainer,
        TrainingConfig,
        JsonlLogger,
    )

    urdf = data / "g1_description/g1_29dof_rev_1_0.urdf"
    usd = data / "g1_usd/g1_29dof/g1_29dof.usda"
    settings = (
        "recovery_enabled",
        "actuator_type",
        "max_delay",
        "num_envs",
        "iterations",
        "buffer_steps",
        "wm_batch",
        "wm_batches",
        "policy_batch",
        "horizon",
        "warmup_steps",
        "critic_epochs",
        "critic_minibatches",
        "save_every",
        "log_every",
    )
    overrides = {
        name: getattr(args, name)
        for name in settings
        if getattr(args, name) is not None
    }
    locations = dict(
        device="cuda:0",
        log_dir=str(args.log_dir),
        urdf=str(urdf),
        robot_usd=str(usd),
        motion_file=str(output),
    )
    if args.resume:
        from mimic_baseline.tasks.supermimic.runtime.configuration import (
            restore_run_config,
            resolve_backend,
        )

        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        args.backend = resolve_backend(args.backend, saved)
        restored, args.small_model = restore_run_config(
            saved, {**overrides, **locations}, small_model=args.small_model
        )
        cfg = TrainingConfig(**restored)
        if args.mode == "train" and (
            args.iterations is None or cfg.iterations <= saved["iteration"]
        ):
            raise ValueError(
                "Resume requires --iterations greater than the saved iteration"
            )
        del saved
    else:
        from mimic_baseline.tasks.supermimic.runtime.configuration import (
            resolve_backend,
        )

        args.backend = resolve_backend(args.backend, None)
        cfg = TrainingConfig(
            robot_name="g1",
            num_envs=args.num_envs or 64,
            iterations=args.iterations or 10,
            perception=False,
            recovery_enabled=True
            if args.recovery_enabled is None
            else args.recovery_enabled,
            actuator_type=args.actuator_type or "delayed_pd",
            max_delay=(0 if args.actuator_type == "implicit" else 4)
            if args.max_delay is None
            else args.max_delay,
            eval_every=0,
            log_every=1,
            **locations,
        )
        cfg.save_every = cfg.iterations
        if args.small_model:
            cfg.buffer_steps = 64
            cfg.warmup_steps = 16
            cfg.wm_batches = 1
            cfg.wm_batch = 32
            cfg.policy_batch = 32
            cfg.horizon = 8
            cfg.critic_epochs = 1
            cfg.critic_minibatches = 1
        for name, value in overrides.items():
            setattr(cfg, name, value)
    if args.mode == "train" and not args.resume and (args.log_dir / "last.pt").exists():
        raise ValueError(
            "Output already contains a checkpoint; use --resume or a new log directory"
        )
    cfg.validate()
    torch.manual_seed(cfg.seed)
    kin_type = {"reference": G1Kinematics, "batched": G1BatchedKinematics,
                "lean": G1LeanKinematics, "bounded": G1TrainingKinematics}[args.kinematics_execution]
    kin = (G1TrainingKinematics(urdf, cfg.device, chunk_size=args.fk_chunk_size)
           if args.kinematics_execution == "bounded" else kin_type(urdf, cfg.device))
    motion = MotionLib(paths, kin, cfg.device)
    if args.mode == "evaluate":
        cfg.num_envs = len(paths)
    mdp = MDP(
        kin,
        motion,
        torch.zeros(29, device=cfg.device),
        recovery_enabled=cfg.recovery_enabled,
    )
    env_cfg_type = G1SuperMimicEnvCfg
    if args.mode == "evaluate":
        from mimic_baseline.tasks.supermimic.envs.evaluation import G1EvaluationEnvCfg

        env_cfg_type = G1EvaluationEnvCfg
    env_cfg = env_cfg_type(
        backend=args.backend,
        self_collisions=cfg.self_collisions,
        recovery_enabled=cfg.recovery_enabled,
        actuator_type=cfg.actuator_type,
        max_delay=cfg.max_delay,
        actor_relative_orientation=cfg.actor_relative_orientation,
        robot_usd=str(usd),
        robot_urdf=str(urdf),
        ground_usd=str(data / "ground/default_ground_plane.usda"),
        eval_mode=args.mode == "evaluate",
    )
    if args.mode == "evaluate":
        env_cfg.kinematics_execution = args.kinematics_execution
        env_cfg.evaluation_execution = args.evaluation_execution
        env_cfg.evaluation_profile = args.evaluation_profile
        env_cfg.evaluation_clip_steps = int(
            (motion.original_clip_end - motion.clip_start).max().item()
        )
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.commands.motion.goal_frames = cfg.goal_frames
    env_cfg.commands.motion.gamma = cfg.gamma
    env_cfg.sim.device = cfg.device
    if args.mode == "check" and not cfg.recovery_enabled:
        env_cfg.episode_length_s = 0.04
        env_cfg.reset_noise = False
    if args.mode in ("train", "check") and args.training_execution == "grouped":
        from mimic_baseline.tasks.supermimic.envs.grouped_actuators import group_g1_actuators
        group_g1_actuators(env_cfg)
    with launch_simulation(env_cfg.sim, {"viz": "none", "device": cfg.device}):
        import gymnasium as gym

        task = (
            "SuperMimic-Recovery-G1-v0"
            if cfg.recovery_enabled
            else "SuperMimic-Flat-G1-v0"
        )
        if args.mode == "evaluate":
            task = "SuperMimic-Evaluation-G1-v0"
        managed = gym.make(task, cfg=env_cfg, motion=motion, mdp=mdp).unwrapped
        env = SuperMimicTrainingEnv(managed)
        try:
            env.reset()
            if args.mode == "check":
                from mimic_baseline.tasks.supermimic.validation.g1 import (
                    check_environment,
                    check_recovery_environment,
                )

                result = (
                    check_recovery_environment(env)
                    if cfg.recovery_enabled
                    else check_environment(env)
                )
                (args.log_dir / "environment.json").write_text(
                    json.dumps(result, indent=2)
                )
                print(json.dumps(result))
                return
            from mimic_baseline.tasks.supermimic.runtime.configuration import (
                build_g1_models,
            )

            wm, policy, critic = build_g1_models(cfg, small_model=args.small_model)
            if args.mode == "evaluate":
                if not args.resume:
                    raise ValueError("Evaluation requires --resume")
                ck = torch.load(
                    args.resume, map_location=cfg.device, weights_only=False
                )
                if ck.get("format") != "supermimic-g1-v1":
                    raise ValueError("Expected ManagerBased G1 checkpoint")
                policy.load_state_dict(ck["policy"])
                policy.eval()
                from mimic_baseline.tasks.supermimic.evaluation import (
                    evaluate_tracking,
                    write_results,
                )

                from mimic_baseline.tasks.supermimic.evaluation.profiles import (
                    runtime_profile,
                )

                audit = runtime_profile(managed, args.evaluation_profile)
                from mimic_baseline.tasks.supermimic.evaluation.progress import (
                    RolloutProgress,
                )

                records = evaluate_tracking(
                    env,
                    motion,
                    mdp,
                    policy,
                    goal_frames=cfg.goal_frames,
                    hist=cfg.hist,
                    use_original_lengths=True,
                    no_termination=args.evaluation_profile == "full_clean",
                    progress_callback=RolloutProgress(args.log_dir),
                )
                audit["after_rollout"] = runtime_profile(
                    managed, args.evaluation_profile
                )
                (args.log_dir / "runtime_profile.json").write_text(
                    json.dumps(audit, indent=2), encoding="utf-8"
                )
                if args.evaluation_profile == "full_clean":
                    from mimic_baseline.tasks.supermimic.evaluation.clean_reporting import (
                        write_clean_results,
                    )

                    write_results = write_clean_results
                write_results(
                    args.log_dir,
                    records,
                    metadata={
                        "evaluation": audit,
                        "robot": "g1",
                        "environment": "SuperMimicRLEnv",
                        "backend": args.backend,
                        "kinematics_execution": args.kinematics_execution,
                    },
                )
                print(
                    json.dumps(
                        {
                            "clips": len(records),
                            (
                                "covered_clips"
                                if args.evaluation_profile == "full_clean"
                                else "completed"
                            ): sum(r["completed"] for r in records),
                        }
                    )
                )
                return
            replay = ReplayBuffer(
                cfg.buffer_steps,
                cfg.num_envs,
                cfg.device,
                action_dim=29,
                recovery_enabled=cfg.recovery_enabled,
            )
            trainer = Trainer(
                env,
                replay,
                motion,
                mdp,
                wm,
                policy,
                critic,
                unflatten_state,
                normalize_xi,
                cfg,
                logger=JsonlLogger(args.log_dir / "train.jsonl"),
                provenance={
                    "environment": "original SuperMimicRLEnv",
                    "robot": "g1",
                    "small_model": args.small_model,
                    "backend": args.backend,
                    "kinematics_execution": args.kinematics_execution,
                    "actuator_execution": args.training_execution,
                    "fk_chunk_size": args.fk_chunk_size if args.kinematics_execution == "bounded" else None,
                },
            )
            if args.resume:
                trainer.load_checkpoint(args.resume)
            trainer.train()
            assert trainer.iteration == cfg.iterations
            for model in (wm, policy, critic):
                assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            print(
                json.dumps({"iteration": trainer.iteration, "finite_parameters": True})
            )
        finally:
            env.close()
