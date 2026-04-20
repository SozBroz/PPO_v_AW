"""
AWBW DRL Bot — Training entry point.

Usage:
  python train.py                              # headless, 6 envs, CUDA, unlimited steps
  python train.py --iters 1000000             # stop after 1M timesteps
  python train.py --n-envs 4                  # 4 parallel game workers
  python train.py --device cpu                # force CPU (no GPU)
  python train.py --map-id 133665             # train on one map only
  python train.py --watch-only                # watch a single random game (debug)
  python train.py --watch-only --map-id 133665 --co-p0 7 --co-p1 1
  python train.py --map-id 123858 --tier T3 --co-p0 1 --co-p1 1 --curriculum-tag misery-andy
  python train.py --n-envs 12 --n-steps 2048 --map-id 123858 --tier T3 --co-p0 1 --co-p1 1
  python train.py --rank                      # compute CO rankings from game log
  python train.py --features                  # compute map features from CSVs
"""
import argparse
import signal
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _install_sigint_first_only() -> None:
    """
    First Ctrl+C stops training; ignore further SIGINT so shutdown (e.g. checkpoint save)
    is not torn down by accidental extra keypresses.
    """

    def _handler(signum: int, frame) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="AWBW DRL Bot")
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Total training timesteps (default: unlimited)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=6,
        help="Parallel game workers for rollout collection (default: 6)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=512,
        help="PPO rollout length per env before each update (default: 512)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help='Torch device: "cuda", "cpu", or "auto" (default: auto)',
    )
    parser.add_argument(
        "--map-id", type=int, default=None,
        help="Train/watch on a specific map ID only",
    )
    parser.add_argument(
        "--watch-only", action="store_true",
        help="Watch a single game with random policies (engine smoke test)",
    )
    parser.add_argument(
        "--rank", action="store_true",
        help="Compute CO rankings from logs/game_log.jsonl",
    )
    parser.add_argument(
        "--features", action="store_true",
        help="Compute map features from CSV files",
    )
    parser.add_argument(
        "--co-p0", type=int, default=None,
        help="Fix P0 CO id (training and watch); default watch-only: 1=Andy",
    )
    parser.add_argument(
        "--co-p1", type=int, default=None,
        help="Fix P1 CO id (training and watch); default watch-only: 7=Max",
    )
    parser.add_argument(
        "--tier", type=str, default=None,
        help='Fixed tier name for training (e.g. T3 for Misery Andy mirror with --co-p0 1 --co-p1 1)',
    )
    parser.add_argument(
        "--curriculum-broad-prob", type=float, default=0.0,
        help="Per-episode probability of full random CO/tier sampling (mixture); 0 = always fixed when set",
    )
    parser.add_argument(
        "--curriculum-tag", type=str, default=None,
        help="Optional label written to game_log rows for slicing TensorBoard / analysis",
    )
    parser.add_argument(
        "--save-every", type=int, default=50_000,
        help="Save checkpoint every N steps (default: 50k)",
    )
    parser.add_argument(
        "--checkpoint-pool", type=int, default=5,
        help="Historical checkpoints to rotate as opponent (default: 5)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.05,
        help="PPO entropy coefficient (default 0.05; narrow Misery-Andy runs often use 0.02)",
    )
    parser.add_argument(
        "--opponent-mix", type=float, default=0.0,
        help="With loaded checkpoint opponent, probability of using capture-greedy instead (0–1)",
    )
    parser.add_argument(
        "--machine-role", type=str, default=None,
        help="Override AWBW_MACHINE_ROLE: main (default) or auxiliary",
    )
    parser.add_argument(
        "--shared-root", type=str, default=None,
        help="Override AWBW_SHARED_ROOT (aux default Z:\\; main must match repo or be unset)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Checkpoint directory (default repo/checkpoints; pool aux: .../checkpoints/pool/<ID>/)",
    )
    parser.add_argument(
        "--pool-from-fleet", action="store_true",
        help="Include checkpoints/pool/*/checkpoint_*.zip in opponent sampling",
    )
    parser.add_argument(
        "--load-promoted", action="store_true",
        help="On startup prefer checkpoints/promoted/best.zip over latest.zip when newer",
    )
    parser.add_argument(
        "--bc-init", type=Path, default=None,
        help="Fresh-run warm-start zip (e.g. checkpoints/bc/bc_warmstart_*.zip); ignored when resuming",
    )
    parser.add_argument(
        "--shared-training", action="store_true",
        help="Reserved for MASTERPLAN §10 async weight sync (not implemented)",
    )
    args = parser.parse_args()

    # ── Resolve device ────────────────────────────────────────────────────────
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    # ── Utility modes ─────────────────────────────────────────────────────────
    if args.features:
        print("[train] Computing map features...")
        from analysis.map_features import compute_all_features
        compute_all_features()
        return

    if args.rank:
        print("[train] Computing CO rankings...")
        from analysis.co_ranker import compute_rankings
        compute_rankings()
        return

    if args.watch_only:
        print("[train] Running engine smoke-test (random policy, single game)...")
        _install_sigint_first_only()
        from rl.self_play import watch_game
        co_p0 = args.co_p0 if args.co_p0 is not None else 1
        co_p1 = args.co_p1 if args.co_p1 is not None else 7
        watch_game(map_id=args.map_id, co_p0=co_p0, co_p1=co_p1)
        return

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"[train] Device  : {device}")
    print(f"[train] Envs    : {args.n_envs} parallel workers")
    print(f"[train] PPO     : n_steps={args.n_steps} (rollout {args.n_steps * args.n_envs:,} env steps/update)")
    print(f"[train] Steps   : {args.iters if args.iters is not None else 'unlimited'}")
    print(f"[train] Map     : {args.map_id or 'all'}")
    if args.tier or args.co_p0 is not None or args.co_p1 is not None:
        print(
            f"[train] Curriculum: tier={args.tier!r} co_p0={args.co_p0} co_p1={args.co_p1} "
            f"broad_prob={args.curriculum_broad_prob} tag={args.curriculum_tag!r}"
        )

    import os as _os
    _env_flags = [
        ("AWBW_TIME_COST", _os.environ.get("AWBW_TIME_COST")),
        ("AWBW_INCOME_TERM_COEF", _os.environ.get("AWBW_INCOME_TERM_COEF")),
        ("AWBW_BUILD_MASK_INFANTRY_ONLY", _os.environ.get("AWBW_BUILD_MASK_INFANTRY_ONLY")),
        ("AWBW_LOG_REPLAY_FRAMES", _os.environ.get("AWBW_LOG_REPLAY_FRAMES")),
    ]
    _active = [f"{k}={v!r}" for k, v in _env_flags if v not in (None, "", "0")]
    if _active:
        print("[train] Env toggles: " + "; ".join(_active))

    if args.shared_training:
        print(
            "[train] --shared-training is reserved (MASTERPLAN §10); "
            "no runtime weight sync in this build."
        )

    from rl.fleet_env import (
        FleetConfig,
        REPO_ROOT,
        bootstrap_fleet_layout,
        load_machine_id,
        load_machine_role,
        load_shared_root_for_role,
        resolve_checkpoint_dir,
        validate_aux_pool_checkpoint_dir,
        validate_fleet_at_startup,
    )

    role = load_machine_role(args.machine_role)
    shared = load_shared_root_for_role(role, args.shared_root)
    fleet_cfg = FleetConfig(
        role=role,
        machine_id=load_machine_id(),
        shared_root=shared,
        repo_root=REPO_ROOT,
    )
    validate_fleet_at_startup(fleet_cfg)
    checkpoint_dir = resolve_checkpoint_dir(REPO_ROOT, args.checkpoint_dir, None)
    validate_aux_pool_checkpoint_dir(fleet_cfg, checkpoint_dir)
    layout_root = fleet_cfg.shared_root if fleet_cfg.is_auxiliary else fleet_cfg.repo_root
    bootstrap_fleet_layout(layout_root, machine_id=fleet_cfg.machine_id, role=fleet_cfg.role)

    from rl.self_play import SelfPlayTrainer
    trainer = SelfPlayTrainer(
        total_timesteps=args.iters,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        device=device,
        save_every=args.save_every,
        checkpoint_pool_size=args.checkpoint_pool,
        map_id=args.map_id,
        co_p0=args.co_p0,
        co_p1=args.co_p1,
        tier_name=args.tier,
        curriculum_broad_prob=args.curriculum_broad_prob,
        curriculum_tag=args.curriculum_tag,
        opponent_mix=args.opponent_mix,
        ent_coef=args.ent_coef,
        checkpoint_dir=checkpoint_dir,
        pool_from_fleet=args.pool_from_fleet,
        load_promoted=args.load_promoted,
        bc_init=args.bc_init,
    )
    _install_sigint_first_only()
    trainer.train()


if __name__ == "__main__":
    # SubprocVecEnv on Windows requires the freeze_support guard
    import multiprocessing
    multiprocessing.freeze_support()
    main()
