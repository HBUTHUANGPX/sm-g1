"""G1 entry point using the ORIGINAL ManagerBased environment and Trainer.

This file only prepares inputs, composes existing components and invokes them.
It contains no substitute simulator loop or learning algorithm.
"""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/mimic_baseline"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--mode", choices=["prepare", "check", "train", "evaluate"], default="check")
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
    import torch
    from mimic_baseline.tasks.supermimic.lafan1 import convert_clip, ensure_clip

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
    from mimic_baseline.tasks.supermimic.g1 import G1Kinematics
    from mimic_baseline.tasks.supermimic.g1_env import G1SuperMimicEnvCfg
    from mimic_baseline.tasks.supermimic.env_managed import SuperMimicRLEnv, SuperMimicTrainingEnv
    from mimic_baseline.tasks.supermimic.motion import MotionLib
    from mimic_baseline.tasks.supermimic.mdp_legacy import MDP, unflatten_state, normalize_xi, INV_DIM, CRITIC_EXTRA
    from mimic_baseline.tasks.supermimic.networks import WorldModel, Policy, Critic
    from mimic_baseline.tasks.supermimic.buffer import ReplayBuffer
    from mimic_baseline.tasks.supermimic.training import Trainer, TrainingConfig, JsonlLogger

    urdf = data / "g1_description/g1_29dof_rev_1_0.urdf"
    usd = data / "g1_usd/g1_29dof/g1_29dof.usda"
    settings = (
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
    overrides = {name: getattr(args, name) for name in settings if getattr(args, name) is not None}
    locations = dict(
        device="cuda:0", log_dir=str(args.log_dir), urdf=str(urdf), robot_usd=str(usd), motion_file=str(output)
    )
    if args.resume:
        from mimic_baseline.tasks.supermimic.g1_run_config import restore_run_config

        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        restored, args.small_model = restore_run_config(saved, {**overrides, **locations}, small_model=args.small_model)
        cfg = TrainingConfig(**restored)
        if args.mode == "train" and (args.iterations is None or cfg.iterations <= saved["iteration"]):
            raise ValueError("Resume requires --iterations greater than the saved iteration")
        del saved
    else:
        cfg = TrainingConfig(
            robot_name="g1",
            num_envs=args.num_envs or 64,
            iterations=args.iterations or 10,
            perception=False,
            actuator_type="implicit",
            max_delay=0,
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
        raise ValueError("Output already contains a checkpoint; use --resume or a new log directory")
    cfg.validate()
    torch.manual_seed(cfg.seed)
    kin = G1Kinematics(urdf, cfg.device)
    motion = MotionLib(paths, kin, cfg.device)
    if args.mode == "evaluate":
        cfg.num_envs = len(paths)
    mdp = MDP(kin, motion, torch.zeros(29, device=cfg.device))
    env_cfg = G1SuperMimicEnvCfg(
        robot_usd=str(usd),
        robot_urdf=str(urdf),
        ground_usd=str(data / "ground/default_ground_plane.usda"),
        eval_mode=args.mode == "evaluate",
    )
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.commands.motion.goal_frames = cfg.goal_frames
    env_cfg.commands.motion.gamma = cfg.gamma
    env_cfg.sim.device = cfg.device
    if args.mode == "check":
        env_cfg.episode_length_s = 0.04
        env_cfg.reset_noise = False
    with launch_simulation(env_cfg.sim, {"viz": "none", "device": cfg.device}):
        import gymnasium as gym

        managed = gym.make("SuperMimic-Flat-G1-v0", cfg=env_cfg, motion=motion, mdp=mdp).unwrapped
        env = SuperMimicTrainingEnv(managed)
        try:
            env.reset()
            if args.mode == "check":
                from mimic_baseline.tasks.supermimic.g1_validation import check_environment

                result = check_environment(env)
                (args.log_dir / "environment.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result))
                return
            kwargs = {"hidden": (64, 64)} if args.small_model else {}
            wm = WorldModel(target_mode=cfg.wm_target_mode, action_dim=29, **kwargs).to(cfg.device)
            policy = Policy(
                93,
                73 * cfg.goal_frames,
                hist=cfg.hist,
                action_dim=29,
                actor_relative_orientation=cfg.actor_relative_orientation,
                **kwargs,
            ).to(cfg.device)
            critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * cfg.goal_frames).to(cfg.device)
            if args.mode == "evaluate":
                if not args.resume:
                    raise ValueError("Evaluation requires --resume")
                ck = torch.load(args.resume, map_location=cfg.device, weights_only=False)
                if ck.get("format") != "supermimic-g1-v1":
                    raise ValueError("Expected ManagerBased G1 checkpoint")
                policy.load_state_dict(ck["policy"])
                policy.eval()
                from mimic_baseline.tasks.supermimic.evaluation import evaluate_tracking, write_results

                records = evaluate_tracking(
                    env, motion, mdp, policy, goal_frames=cfg.goal_frames, hist=cfg.hist, use_original_lengths=True
                )
                write_results(args.log_dir, records, metadata={"robot": "g1", "environment": "SuperMimicRLEnv"})
                print(json.dumps({"clips": len(records), "completed": sum(r["completed"] for r in records)}))
                return
            replay = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, cfg.device, action_dim=29)
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
                provenance={"environment": "original SuperMimicRLEnv", "robot": "g1", "small_model": args.small_model},
            )
            if args.resume:
                trainer.load_checkpoint(args.resume)
            trainer.train()
            assert trainer.iteration == cfg.iterations
            for model in (wm, policy, critic):
                assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            print(json.dumps({"iteration": trainer.iteration, "finite_parameters": True}))
        finally:
            env.close()


if __name__ == "__main__":
    main()
