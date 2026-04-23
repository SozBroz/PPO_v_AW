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
import os
import signal
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _load_dotenv(path: Path) -> None:
    """Optional repo-root .env: KEY=VALUE lines; does not override existing env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def build_train_argument_parser() -> argparse.ArgumentParser:
    """CLI parser for ``train.py`` (also used by ``rl.ai_vs_ai`` to mirror a live run)."""
    parser = argparse.ArgumentParser(description="AWBW DRL Bot")
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Total training timesteps (default: unlimited)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=6,
        help=(
            "Parallel SubprocVecEnv game workers (default: 6). "
            "More workers raise throughput (steps/s) but cost ~2-3 GB host RAM each "
            "and keep the step loop synchronous — every step waits for the slowest env. "
            "Scaling tip: if GPU utilization is low and host RAM has headroom, "
            "raising n_envs is the most effective throughput lever."
        ),
    )
    parser.add_argument(
        "--n-steps", type=int, default=512,
        help=(
            "PPO rollout length per env before each update (default: 512). "
            "Increasing gives longer on-policy trajectories (can improve credit assignment) "
            "at the cost of more VRAM (rollout buffer grows linearly). "
            "Scaling tip: safe to raise if n_steps * n_envs still fits in VRAM after --batch-size is tuned."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help=(
            "PPO minibatch size (default: 256; must be <= n_steps * n_envs). "
            "Larger batches = more stable gradient estimates; "
            "can raise up to n_steps * n_envs for full-batch PPO "
            "as long as it fits in VRAM. Tuning tip: this is the lowest-VRAM-cost "
            "knob to push toward the rollout cap."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help=(
            'Torch device for the PPO learner: "cuda", "cpu", or "auto" (default: auto). '
            "Opponent inference always runs on CPU regardless of this flag. "
            "VRAM is consumed by the policy network + optimizer + rollout buffer "
            "(scales with n_steps * n_envs * obs_shape). "
            "If VRAM is tight, reduce --batch-size first (smallest impact on sample efficiency), "
            "then --n-steps; reducing --n_envs also helps but lowers throughput."
        ),
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
        "--checkpoint-zip-cap",
        type=int,
        default=100,
        help=(
            "Max on-disk checkpoint_*.zip files under checkpoint-dir (oldest by mtime "
            "deleted after each save). Use 0 for unlimited."
        ),
    )
    parser.add_argument(
        "--checkpoint-curate",
        action="store_true",
        default=False,
        help=(
            "Phase 10b: replace FIFO-by-mtime pruning with a curator that "
            "keeps K newest + M top-by-verdict-winrate + D diversity slots. "
            "Default off; falls back to FIFO when no verdicts available."
        ),
    )
    parser.add_argument("--curator-k-newest", type=int, default=8)
    parser.add_argument("--curator-m-top-winrate", type=int, default=12)
    parser.add_argument("--curator-d-diversity", type=int, default=4)
    parser.add_argument("--curator-min-age-minutes", type=float, default=5.0)
    parser.add_argument(
        "--verdicts-root",
        type=str,
        default=None,
        help=(
            "Phase 10b: shared <root>/fleet/ directory the curator reads "
            "verdict json from. None = curator falls back to FIFO."
        ),
    )
    parser.add_argument(
        "--local-checkpoint-mirror",
        type=str,
        default=None,
        help=(
            "Phase 10a: local fast-disk directory the trainer writes checkpoints "
            "to first; a background publisher copies them to the shared "
            "checkpoint dir asynchronously. Default off (legacy direct-write "
            "semantics). Recommended on aux pool trainers writing to slow "
            "Samba/HDD shares — e.g. C:\\Users\\<you>\\.awbw_local_ckpt\\<id> "
            "on Windows."
        ),
    )
    parser.add_argument(
        "--publisher-queue-max",
        type=int,
        default=4,
        help="Phase 10a: max queued async publishes when --local-checkpoint-mirror is set (default 4).",
    )
    parser.add_argument(
        "--publisher-drain-timeout-s",
        type=float,
        default=60.0,
        help="Phase 10a: max seconds to wait for the publisher on shutdown (default 60).",
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
        "--pool-from-fleet",
        action="store_true",
        help=(
            "Fleet opponent pool: merge checkpoint_*.zip from the fleet checkpoints root "
            "(top-level + checkpoints/pool/*/). On auxiliary pool trainers the root is "
            "the shared checkpoints/ (e.g. Z:\\checkpoints), not only the pool/<ID>/ leaf."
        ),
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
    parser.add_argument(
        "--cold-opponent", type=str, default="random",
        choices=("random", "greedy_capture", "end_turn"),
        help=(
            "Cold-start opponent (no checkpoints loaded yet). "
            "'random' (default): uniform random legal action — gives the learner "
            "a chance to discover capture before facing a teacher. "
            "'greedy_capture': pre-fix legacy default; aggressive bootstrap. "
            "'end_turn': punching bag — picks END_TURN whenever legal. "
            "Used for the smoke gate in plan p0-capture-architecture-fix."
        ),
    )
    parser.add_argument(
        "--opponent-refresh-rollouts",
        type=int,
        default=4,
        help=(
            "Phase 10c: refresh opponent pool every N rollouts (vec_env."
            "env_method). 0 disables. Default 4."
        ),
    )
    parser.add_argument(
        "--hot-reload-enabled",
        action="store_true",
        default=False,
        help=(
            "Phase 10d: when set, watch <shared>/fleet/<id>/reload_request.json "
            "and apply target weights at rollout boundary. Default OFF — "
            "orchestrator (Phase 10e) flips this per-machine."
        ),
    )
    parser.add_argument(
        "--hot-reload-min-steps-done",
        type=int,
        default=0,
        help="Phase 10d: minimum self.steps_done before honoring a reload request.",
    )
    parser.add_argument(
        "--learner-greedy-mix", type=float, default=0.0,
        help=(
            "Probability that the learner action is overridden by the same "
            "capture-greedy heuristic that bootstraps the opponent (DAGGER-lite). "
            "Set in (0, 0.5] for early training, then restart at 0 to decay. "
            "Sets AWBW_LEARNER_GREEDY_MIX env var; SubprocVecEnv workers inherit."
        ),
    )
    parser.add_argument(
        "--capture-move-gate", action="store_true",
        help=(
            "Restrict infantry/mech MOVE choices to capturable enemy/neutral "
            "property tiles whenever any are reachable. Closes the "
            "SELECT-MOVE-WAIT-in-place loophole. Sets AWBW_CAPTURE_MOVE_GATE=1; "
            "SubprocVecEnv workers inherit. Engine step() bypasses the mask, "
            "so replays are unaffected."
        ),
    )
    return parser


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
    _load_dotenv(ROOT / ".env")
    parser = build_train_argument_parser()
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
    print(
        f"[train] PPO     : n_steps={args.n_steps} batch_size={args.batch_size} "
        f"(rollout {args.n_steps * args.n_envs:,} env steps/update)"
    )
    print(f"[train] Steps   : {args.iters if args.iters is not None else 'unlimited'}")
    print(f"[train] Map     : {args.map_id or 'all'}")
    if args.tier or args.co_p0 is not None or args.co_p1 is not None:
        print(
            f"[train] Curriculum: tier={args.tier!r} co_p0={args.co_p0} co_p1={args.co_p1} "
            f"broad_prob={args.curriculum_broad_prob} tag={args.curriculum_tag!r}"
        )

    if args.learner_greedy_mix and args.learner_greedy_mix > 0.0:
        os.environ["AWBW_LEARNER_GREEDY_MIX"] = str(float(args.learner_greedy_mix))

    if args.capture_move_gate:
        os.environ["AWBW_CAPTURE_MOVE_GATE"] = "1"

    _env_flags = [
        ("AWBW_TIME_COST", os.environ.get("AWBW_TIME_COST")),
        ("AWBW_INCOME_TERM_COEF", os.environ.get("AWBW_INCOME_TERM_COEF")),
        ("AWBW_BUILD_MASK_INFANTRY_ONLY", os.environ.get("AWBW_BUILD_MASK_INFANTRY_ONLY")),
        ("AWBW_LOG_REPLAY_FRAMES", os.environ.get("AWBW_LOG_REPLAY_FRAMES")),
        ("AWBW_LEARNER_GREEDY_MIX", os.environ.get("AWBW_LEARNER_GREEDY_MIX")),
        ("AWBW_CAPTURE_MOVE_GATE", os.environ.get("AWBW_CAPTURE_MOVE_GATE")),
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
        resolve_fleet_opponent_pool_root,
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

    fleet_opponent_root = (
        str(resolve_fleet_opponent_pool_root(checkpoint_dir, fleet_cfg))
        if args.pool_from_fleet
        else None
    )

    from rl.self_play import SelfPlayTrainer
    trainer = SelfPlayTrainer(
        total_timesteps=args.iters,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
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
        fleet_opponent_root=fleet_opponent_root,
        checkpoint_zip_cap=args.checkpoint_zip_cap,
        checkpoint_curate=args.checkpoint_curate,
        curator_k_newest=args.curator_k_newest,
        curator_m_top_winrate=args.curator_m_top_winrate,
        curator_d_diversity=args.curator_d_diversity,
        curator_min_age_minutes=args.curator_min_age_minutes,
        verdicts_root=args.verdicts_root,
        load_promoted=args.load_promoted,
        bc_init=args.bc_init,
        cold_opponent=args.cold_opponent,
        local_checkpoint_mirror=args.local_checkpoint_mirror,
        publisher_queue_max=args.publisher_queue_max,
        publisher_drain_timeout_s=args.publisher_drain_timeout_s,
        fleet_cfg=fleet_cfg,
        opponent_refresh_rollouts=args.opponent_refresh_rollouts,
        hot_reload_enabled=args.hot_reload_enabled,
        hot_reload_min_steps_done=args.hot_reload_min_steps_done,
    )
    _install_sigint_first_only()
    trainer.train()


if __name__ == "__main__":
    # SubprocVecEnv on Windows requires the freeze_support guard
    import multiprocessing
    multiprocessing.freeze_support()
    main()
