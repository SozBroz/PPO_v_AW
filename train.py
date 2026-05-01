"""
AWBW DRL Bot — Training entry point.

Usage:
  python train.py                              # headless, 14 envs, CUDA, unlimited steps
  python train.py --iters 1000000             # stop after 1M timesteps
  python train.py --n-envs 4                  # 4 parallel game workers
  python train.py --device cpu                # force CPU (no GPU)
  python train.py --map-id 133665             # train on one map only
  python train.py --map-id 123858,133665     # random map each episode from this list
  python train.py --co-p0 1,14 --co-p1 12    # random CO per seat from lists (with --tier)
  python train.py --map-id std               # GL std map pool (random; same as omitting --map-id)
  python train.py --watch-only                # watch a single random game (debug)
  python train.py --watch-only --map-id 133665 --co-p0 7 --co-p1 1
  python train.py --stage1-narrow
  python train.py --map-id 123858 --tier T3 --co-p0 1 --co-p1 1 --curriculum-tag misery-andy
  python train.py --n-envs 12 --n-steps 2048 --map-id 123858 --tier T3 --co-p0 1 --co-p1 1
  python train.py --log-replay-frames         # game_log rows include frames for /replay/
  python train.py --machine-id pc-b           # stamp game_log machine_id (fleet parity)
  python train.py --fps-diag                  # AWBW_FPS_DIAG=1; fps_diag.jsonl (+ async [fps_diag] stdout)
  python train.py --rank                      # compute CO rankings from game log
  python train.py --features                  # compute map features from CSVs
"""
import argparse
import os
import random
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _parse_map_id_cli(value: str) -> list[int] | None:
    """
    ``--map-id`` as comma-separated non-negative ints, a single int, or ``std`` / ``gl-std``
    (case-insensitive) for the full Global League **std** pool — same as omitting ``--map-id``.
    """
    s = (value or "").strip()
    if s.lower() in ("std", "gl-std"):
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            f"invalid --map-id {value!r}: use non-negative id(s), comma-separated, "
            f"or 'std' / 'gl-std'"
        )
    out: list[int] = []
    for p in parts:
        if p.lower() in ("std", "gl-std"):
            raise argparse.ArgumentTypeError(
                f"invalid --map-id segment {p!r}: use 'std' alone for the full pool, not in a list"
            )
        try:
            n = int(p, 10)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"invalid --map-id token {p!r}: expected non-negative integer"
            ) from e
        if n < 0:
            raise argparse.ArgumentTypeError(f"--map-id must be non-negative, got {n}")
        if n not in out:
            out.append(n)
    return out


def _parse_co_csv_cli(value: str) -> list[int]:
    """``--co-p0`` / ``--co-p1`` as comma-separated CO ids (order preserved, duplicates dropped)."""
    s = (value or "").strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("CO list must contain at least one integer id")
    out: list[int] = []
    for p in parts:
        try:
            n = int(p, 10)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"invalid CO id token {p!r}: expected integer"
            ) from e
        if n not in out:
            out.append(n)
    return out


def _parse_capture_move_gate_probability_cli(value: str) -> float:
    """``--capture-move-gate P`` with P in [0, 1] (fractional probability)."""
    try:
        v = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid probability: {value!r}") from e
    if v < 0.0 or v > 1.0:
        raise argparse.ArgumentTypeError(
            f"--capture-move-gate expects P in [0, 1], got {value!r}"
        )
    return v


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


def _print_native_implementation_diag() -> None:
    """
    One-line report of which Cython extension paths are active in this process.
    Aligns with ``rl.encoder`` (``CYTHON_AVAILABLE`` + ``AWBW_USE_CYTHON_ENCODER``)
    and ``engine`` action/occupancy extension modules.
    """
    try:
        from rl import encoder as _encoder_mod
    except Exception as exc:  # pragma: no cover — training cannot run without encoder
        print(f"[train] native: encoder=unavailable ({exc!r})")
        return
    use_enc = os.environ.get("AWBW_USE_CYTHON_ENCODER", "1") == "1" and getattr(
        _encoder_mod, "CYTHON_AVAILABLE", False
    )
    enc = "cython" if use_enc else "python"
    try:
        import engine._action_cython as _action_cython_mod
        import engine._occupancy_cython as _occ_mod
    except Exception as exc:  # pragma: no cover
        print(
            f"[train] native: encoder={enc}, action=unavailable, occupancy=unavailable ({exc!r})"
        )
        return

    def _ext_label(mod) -> str:
        path = (getattr(mod, "__file__", None) or "").lower()
        if path.endswith((".pyd", ".so")):
            return "cython"
        if path.endswith(".py"):
            return "python"
        return "cython" if path else "unknown"

    act = _ext_label(_action_cython_mod)
    occ = _ext_label(_occ_mod)
    print(f"[train] native: encoder={enc}, action={act}, occupancy={occ}")


class _MaxCalendarDaysAction(argparse.Action):
    """``--max-days`` is canonical; ``--max-turns`` warns and sets the same cap."""

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == "--max-turns":
            print(
                "[train] --max-turns is deprecated; use --max-days "
                "(same end-inclusive calendar cap).",
                file=sys.stderr,
            )
        setattr(namespace, self.dest, values)


def _apply_stage1_narrow_defaults(args: argparse.Namespace) -> None:
    """Fill Phase 1a narrow fields when ``--stage1-narrow`` is set (unset slots only)."""
    if not getattr(args, "stage1_narrow", False):
        return
    if args.map_id is None:
        args.map_id = [123858]
    if args.tier is None:
        args.tier = "T3"
    if args.co_p0 is None:
        args.co_p0 = [1]
    if args.co_p1 is None:
        args.co_p1 = [1]
    if args.curriculum_tag is None:
        args.curriculum_tag = "stage1-misery-andy"
    print(
        "[train] --stage1-narrow: map_id=123858 tier=T3 co_p0=1 co_p1=1 "
        "tag=stage1-misery-andy (unset fields only)"
    )


def _sync_worker_inherited_env_flags(args: argparse.Namespace) -> None:
    """
    Subproc / async actors read ``AWBW_LEARNER_GREEDY_MIX``, ``AWBW_CAPTURE_MOVE_GATE``,
    ``AWBW_EGOCENTRIC_EPISODE_PROB``, and ``AWBW_PAIRWISE_ZERO_SUM_REWARD`` from this
    process environment.  After
    :func:`rl.train_launch_env.pop_train_cli_owned_keys_from_os_environ`
    and stripped parent env at spawn, only this function (from parsed CLI) repopulates them.
    """
    if float(args.learner_greedy_mix) > 0.0:
        os.environ["AWBW_LEARNER_GREEDY_MIX"] = str(float(args.learner_greedy_mix))
    else:
        os.environ.pop("AWBW_LEARNER_GREEDY_MIX", None)

    if float(args.egocentric_episode_prob) > 0.0:
        os.environ["AWBW_EGOCENTRIC_EPISODE_PROB"] = str(
            float(args.egocentric_episode_prob)
        )
    else:
        os.environ.pop("AWBW_EGOCENTRIC_EPISODE_PROB", None)

    cmg = float(getattr(args, "capture_move_gate", 0.0) or 0.0)
    if cmg > 0.0:
        os.environ["AWBW_CAPTURE_MOVE_GATE"] = str(cmg)
    else:
        os.environ.pop("AWBW_CAPTURE_MOVE_GATE", None)

    # Deprecated: the extra per-property-loss punishment duplicated Φ economy
    # loss and was large enough to dominate terminal scale. Keep stale shells
    # and old launch overlays from re-enabling it.
    os.environ.pop("AWBW_PHI_ENEMY_PROPERTY_CAPTURE_PENALTY", None)

    if bool(args.pairwise_zero_sum_reward):
        os.environ["AWBW_PAIRWISE_ZERO_SUM_REWARD"] = "1"
    else:
        os.environ.pop("AWBW_PAIRWISE_ZERO_SUM_REWARD", None)


def build_train_argument_parser() -> argparse.ArgumentParser:
    """CLI parser for ``train.py`` (also used by ``rl.ai_vs_ai`` to mirror a live run)."""
    parser = argparse.ArgumentParser(description="AWBW DRL Bot")
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Total training timesteps (default: unlimited)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=14,
        help=(
            "Parallel game workers: SubprocVecEnv (sync) or IMPALA actors (async). "
            "Default 14 matches a typical high-throughput desktop; each process holds "
            "env + policy (~2-3 GiB RSS order-of-magnitude). Expect a short-lived RAM "
            "commit spike at spawn (queue + parallel loads); steady state is usually lower. "
            "Sync: every step waits for the slowest env. Async: raise page file if Windows "
            "kills children during the initial wave. Lower with --n-envs if host RAM is tight."
        ),
    )
    parser.add_argument(
        "--n-steps", type=int, default=512,
        help=(
            "PPO rollout length per env before each update (default: 512). "
            "Increasing gives longer on-policy trajectories (can improve credit assignment) "
            "at the cost of more VRAM (rollout buffer grows linearly). "
            "Scaling tip: safe to raise if n_steps * n_envs still fits in VRAM after --batch-size is tuned. "
            "With larger NN (MASTERPLAN §12.1c), consider n_steps=1024 to amortize PPO update cost."
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
        "--map-id",
        type=_parse_map_id_cli,
        default=None,
        help=(
            "Map id(s): non-negative int or comma-separated list (uniform random each episode). "
            "Use 'std' / 'gl-std' alone for the full GL std pool (same as omitting this flag)."
        ),
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
        "--co-p0", type=_parse_co_csv_cli, default=None,
        help=(
            "P0 CO id(s): comma-separated list (uniform random each episode) or single id. "
            "Omitted → random CO from the pinned tier roster each episode when --tier is set; "
            "otherwise tier/co sampling follows map pool rules. Watch-only defaults elsewhere."
        ),
    )
    parser.add_argument(
        "--co-p1", type=_parse_co_csv_cli, default=None,
        help=(
            "P1 CO id(s): same semantics as --co-p0. "
            "Omitted with --tier draws uniformly from that tier's roster per episode."
        ),
    )
    parser.add_argument(
        "--tier", type=str, default=None,
        help=(
            "Pinned GL tier name per sampled map (e.g. T3). Roster is used even if that row "
            "is disabled on the map JSON; omit CO flags to sample random COs from that roster."
        ),
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
        "--stage1-narrow",
        action="store_true",
        help=(
            "Phase 1a preset: Misery map 123858, tier T3, Andy mirror (co 1 vs 1), "
            "curriculum_tag stage1-misery-andy — only for args left at default None "
            "(override any piece by setting --map-id / --tier / --co-p0 / --co-p1 / "
            "--curriculum-tag explicitly)."
        ),
    )
    parser.add_argument(
        "--save-every", type=int, default=50_000,
        help=(
            "Save checkpoint_<utc>.zip every N env steps (default: 50k). "
            "By default --publish-latest-each-save also overwrites latest.zip each tick."
        ),
    )
    parser.add_argument(
        "--publish-latest-each-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After each timed checkpoint_*.zip save, overwrite latest.zip (default: on). "
            "Use --no-publish-latest-each-save to cut shared-disk writes; latest still saves on exit."
        ),
    )
    parser.add_argument(
        "--checkpoint-pool", type=int, default=24,
        help=(
            "Opponent samples only among the K newest checkpoint_*.zip files "
            "(local + fleet when --pool-from-fleet). Use 0 for no cap (all on disk). "
            "Default: 24."
        ),
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
        "--machine-id",
        type=str,
        default=None,
        help=(
            "Stamp AWBW_MACHINE_ID for this process (and SubprocVecEnv workers) before "
            "training starts. Matches fleet train_launch_cmd.json env; use when running "
            "train.py outside start_solo_training so game_log.jsonl rows carry machine_id."
        ),
    )
    parser.add_argument(
        "--log-replay-frames",
        action="store_true",
        help=(
            "Set AWBW_LOG_REPLAY_FRAMES=1 so each game_log row includes a frames[] array "
            "for the in-repo /replay/ viewer (large logs; default off)."
        ),
    )
    parser.add_argument(
        "--fps-diag",
        action="store_true",
        help=(
            "Set AWBW_FPS_DIAG=1. Sync: SubprocVecEnv worker step stats + "
            "rl/self_play diagnostics to logs/fps_diag.jsonl. Async: same file + "
            "[fps_diag] stdout each learner step (collect vs learner wall split). "
            "NN scalars (loss / policy / value) also go to logs/nn_train.jsonl every learner step; "
            "async fps_diag rows and [fps_diag] lines include the same loss fields. "
            "Workers inherit AWBW_FPS_DIAG when AWBW_TRACK_PER_WORKER_TIMES is not forced off."
        ),
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
        "--training-backend",
        type=str,
        default="sync",
        choices=("sync", "async"),
        help=(
            "sync (default): SubprocVecEnv + MaskablePPO. "
            "async: IMPALA-style parallel env actors + V-trace learner (decoupled stepping)."
        ),
    )
    parser.add_argument(
        "--async-unroll-length",
        type=int,
        default=None,
        help=(
            "async only: env transitions per actor chunk sent to the learner "
            "(default: same as --n-steps)."
        ),
    )
    parser.add_argument(
        "--async-learner-batch",
        type=int,
        default=None,
        help=(
            "async only: transitions per learner update (multiple of unroll). "
            "Default when omitted: min(--n-steps * --n-envs, "
            "max(unroll, AWBW_ASYNC_LEARNER_TRANSITIONS_CAP or 2048), "
            "--n-envs * unroll) snapped to a multiple of unroll (reduces actor queue stalls). "
            "CUDA also microbatches evaluate_actions (see --async-learner-forward-chunk)."
        ),
    )
    parser.add_argument(
        "--async-queue-max",
        type=int,
        default=64,
        help="async only: max queued actor chunks (backpressure when the learner falls behind).",
    )
    parser.add_argument(
        "--async-gpu-opponent-permits-subtract",
        type=int,
        default=2,
        help=(
            "async only, when AWBW_GPU_OPPONENT_POOL is on: reduce concurrent CUDA opponent "
            "forwards vs hybrid pool size (default 2; floor 1 permit). Frees VRAM for the "
            "learner on ~12GB GPUs. Use 0 for full pool size."
        ),
    )
    parser.add_argument(
        "--async-learner-forward-chunk",
        type=int,
        default=None,
        help=(
            "async only, CUDA: observations per evaluate_actions chunk (default 256 via "
            "AWBW_ASYNC_LEARNER_FORWARD_CHUNK; lowers peak learner VRAM). "
            "0 or omitted uses env/default. CPU: ignored (single forward)."
        ),
    )
    parser.add_argument(
        "--cold-opponent", type=str, default="random",
        choices=("random", "greedy_capture", "greedy_mix", "end_turn"),
        help=(
            "Cold-start opponent (no checkpoints loaded yet). "
            "'random' (default): uniform random legal action — gives the learner "
            "a chance to discover capture before facing a teacher. "
            "'greedy_capture': pre-fix legacy default; aggressive bootstrap. "
            "'greedy_mix': half capture-greedy / half random per P1 microstep "
            "(curriculum stage_b+). "
            "'end_turn': punching bag — picks END_TURN whenever legal. "
            "Used for the smoke gate in plan p0-capture-architecture-fix."
        ),
    )
    parser.add_argument(
        "--opening-book", type=Path, default=None,
        help=(
            "JSONL from tools/build_opening_book.py. Applied inside AWBWEnv so "
            "P0 and/or P1 can use it during the opening."
        ),
    )
    parser.add_argument(
        "--opening-book-seat",
        type=int,
        default=1,
        help="Deprecated compatibility alias; prefer --opening-book-seats.",
    )
    parser.add_argument(
        "--opening-book-seats",
        type=str,
        default="both",
        choices=("both", "p0", "p1", "0", "1", "none"),
        help="Which engine seats may use opening books. Default: both.",
    )
    parser.add_argument(
        "--opening-book-prob", type=float, default=1.0,
        help="Per episode probability of enabling the selected opening book lines (0-1).",
    )
    parser.add_argument(
        "--opening-book-strict-co", action=argparse.BooleanOptionalAction, default=False,
        help="Only use books whose co_id matches the live seat CO (if co_id set in book).",
    )
    parser.add_argument(
        "--opening-book-days", type=int, default=0,
        help=(
            "Max engine calendar day for book lines (0 = no day cap; use with short "
            "truncated traces + action-list books from build_opening_book)."
        ),
    )
    parser.add_argument("--opening-book-seed", type=int, default=0, help="RNG seed for book pick / prob.")
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
            "Sets AWBW_LEARNER_GREEDY_MIX for workers; 0 removes it so parent-shell "
            "values cannot override curriculum."
        ),
    )
    parser.add_argument(
        "--egocentric-episode-prob",
        type=float,
        default=0.0,
        help=(
            "Per-episode probability (0–1) to randomize learner seat {0,1} on each "
            "reset; otherwise use AWBW_LEARNER_SEAT / default 0. Live snapshot workers "
            "keep pinned seats. Sets AWBW_EGOCENTRIC_EPISODE_PROB; 0 removes it."
        ),
    )
    parser.add_argument(
        "--dual-gradient-self-play",
        action="store_true",
        help=(
            "Async-only: both engine seats sample from the shared policy and each "
            "active-seat decision is recorded as a policy-gradient row with "
            "seat-relative zero-sum Phi/reward signals. Requires --training-backend async."
        ),
    )
    parser.add_argument(
        "--dual-gradient-hist-prob",
        type=float,
        default=0.0,
        help=(
            "Only with async --dual-gradient-self-play: probability each episode uses a "
            "historical checkpoint as the opponent (standard env.step rollout) instead "
            "of symmetric mirror self-play from synced weights (1 − this = mirror-SP "
            'fraction). Set to 0.2 for "~80%% mirror / 20%% vs archive".'
        ),
    )
    parser.add_argument(
        "--capture-move-gate",
        nargs="?",
        const=1.0,
        default=0.0,
        metavar="P",
        type=_parse_capture_move_gate_probability_cli,
        help=(
            "Infantry/mech MOVE gate near capturable properties: probability P in "
            "[0,1] that reachable MOVE destinations are restricted to capturable tiles "
            "(closes SELECT-MOVE-WAIT in place). 0 disables. Omit the value for P=1. "
            "For 0<P<1 the trial is deterministic from game state+capturable set "
            "(does not consume combat RNG; redundant legal checks agree). Sets "
            "AWBW_CAPTURE_MOVE_GATE for workers."
        ),
    )
    parser.add_argument(
        "--pairwise-zero-sum-reward",
        action="store_true",
        help=(
            "Opt in to the learner-frame pairwise reward contract for AWBWEnv.step(): "
            "competitive reward is exposed as a zero-sum seat pair while draw/time "
            "and discipline penalties stay explicit. Does not alter "
            "step_active_seat_once / dual-gradient active-seat rewards."
        ),
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=10000,
        help=(
            "Hard cap on P0 env.step calls per episode; episode ends with "
            "truncated=True when reached without a natural terminal. "
            "0 or negative disables (not recommended for production)."
        ),
    )
    parser.add_argument(
        "--max-p1-microsteps",
        type=int,
        default=4000,
        help=(
            "Hard cap on opponent microsteps per opponent turn; truncates mid-turn "
            "if exceeded. 0 or negative disables the explicit cap (env still derives "
            "a cap from max_env_steps when that is set)."
        ),
    )
    parser.add_argument(
        "--max-days",
        "--max-turns",
        dest="max_days",
        type=int,
        default=None,
        metavar="N",
        action=_MaxCalendarDaysAction,
        help=(
            "End-inclusive engine calendar day cap (``GameState.turn`` / property tiebreak; "
            "play days 1..N). Omit for engine default (100). Alias ``--max-turns`` (deprecated)."
        ),
    )
    parser.add_argument(
        "--cop-disable-per-seat-p",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Per curriculum episode, each seat that has a COP independently disables "
            "COP activation with this probability in [0,1] (SCOP unchanged). Omit to read "
            "AWBW_COP_DISABLE_PER_SEAT_P (default off). Ignored for live snapshot loads."
        ),
    )
    parser.add_argument(
        "--mcts-mode",
        type=str,
        default="off",
        choices=("off", "eval_only"),
        help=(
            "Phase 11c: MCTS knob storage for orchestration. Default off. "
            "Does not change training rollouts (PPO still uses the policy directly). "
            "MCTS in eval_only mode does not affect training; it only runs in "
            "scripts/symmetric_checkpoint_eval.py for promotion gating."
        ),
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=16,
        help="Phase 11c: MCTS simulations per root (eval_only in symmetric_checkpoint_eval).",
    )
    parser.add_argument(
        "--mcts-c-puct",
        type=float,
        default=1.5,
        help="Phase 11c: PUCT exploration constant.",
    )
    parser.add_argument(
        "--mcts-dirichlet-alpha",
        type=float,
        default=0.3,
        help="Phase 11c: Dirichlet noise alpha at root.",
    )
    parser.add_argument(
        "--mcts-dirichlet-epsilon",
        type=float,
        default=0.25,
        help="Phase 11c: Mixing weight for Dirichlet noise at root.",
    )
    parser.add_argument(
        "--mcts-temperature",
        type=float,
        default=1.0,
        help="Phase 11c: Temperature for final plan selection at root.",
    )
    parser.add_argument(
        "--mcts-min-depth",
        type=int,
        default=4,
        help="Phase 11c: Min tree depth before PUCT (greedy prior above).",
    )
    parser.add_argument(
        "--mcts-root-plans",
        type=int,
        default=8,
        help="Phase 11c: Number of distinct full-turn plans sampled at each expansion.",
    )
    parser.add_argument(
        "--mcts-max-plan-actions",
        type=int,
        default=256,
        help="Phase 11c: Cap on actions per simulated turn rollout.",
    )
    parser.add_argument(
        "--live-games-id",
        type=int,
        action="append",
        default=None,
        help=(
            "In-progress Amarriner game id(s) for live PPO: first N Subproc envs load "
            ".pkl from --live-snapshot-dir (use tools/amarriner_write_live_snapshot.py). "
            "Requires --n-envs >= number of ids."
        ),
    )
    parser.add_argument(
        "--live-learner-seats",
        type=str,
        default=None,
        help="Comma-separated engine seats (0/1) for each --live-games-id; default all 0.",
    )
    parser.add_argument(
        "--live-snapshot-dir",
        type=Path,
        default=None,
        help="Directory containing {games_id}.pkl snapshots (default: .tmp/awbw_live_snapshot).",
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
    from rl import _win_triton_warnings

    _win_triton_warnings.apply()

    _load_dotenv(ROOT / ".env")
    from rl.train_launch_env import pop_train_cli_owned_keys_from_os_environ

    pop_train_cli_owned_keys_from_os_environ()
    # Linux: can reduce allocator fragmentation. Windows PyTorch builds often omit this
    # (warning if set); skip there.
    if os.name != "nt":
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
    parser = build_train_argument_parser()
    args = parser.parse_args()
    _apply_stage1_narrow_defaults(args)
    _print_native_implementation_diag()

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

        def _watch_co(ac: list[int] | None, default: int) -> int:
            if ac is None:
                return default
            if len(ac) == 1:
                return ac[0]
            return int(random.choice(ac))

        mid_w: int | None
        if args.map_id is None:
            mid_w = None
        elif len(args.map_id) == 1:
            mid_w = args.map_id[0]
        else:
            mid_w = int(random.choice(args.map_id))
        watch_game(
            map_id=mid_w,
            co_p0=_watch_co(args.co_p0, 1),
            co_p1=_watch_co(args.co_p1, 7),
        )
        return

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"[train] Device  : {device}")
    print(f"[train] Backend : {args.training_backend}")
    print(f"[train] Envs    : {args.n_envs} parallel workers")
    if args.training_backend == "async":
        from rl.self_play import _snap_async_learner_batch_to_unroll

        _unroll = int(args.async_unroll_length) if args.async_unroll_length is not None else int(args.n_steps)
        _unroll = max(4, _unroll)
        _roll = int(args.n_steps) * int(args.n_envs)
        if args.async_learner_batch is not None:
            _lb = int(args.async_learner_batch)
        else:
            try:
                _cap = int((os.environ.get("AWBW_ASYNC_LEARNER_TRANSITIONS_CAP") or "2048").strip())
            except ValueError:
                _cap = 2048
            _cap = max(64, _cap)
            _one_wave = min(_roll, int(args.n_envs) * _unroll)
            _lb = min(_roll, max(_unroll, _cap), _one_wave)
        _lb = _snap_async_learner_batch_to_unroll(_lb, _unroll, _roll)
        _lb_note = (
            ""
            if args.async_learner_batch is not None or _lb >= _roll
            else ", default VRAM cap"
        )
        print(
            f"[train] Async   : IMPALA actors | unroll={_unroll} | "
            f"learner_batch~{_lb} (rollout={_roll}{_lb_note}) | queue_max={args.async_queue_max} | "
            f"gpu_opp_permits_subtract={args.async_gpu_opponent_permits_subtract}"
        )
    print(
        f"[train] PPO     : n_steps={args.n_steps} batch_size={args.batch_size} "
        f"(rollout {args.n_steps * args.n_envs:,} env steps/update)"
    )
    print(f"[train] Steps   : {args.iters if args.iters is not None else 'unlimited'}")
    _map_line = (
        "std (GL pool)"
        if args.map_id is None
        else (
            str(args.map_id[0])
            if len(args.map_id) == 1
            else f"{len(args.map_id)} maps [{','.join(str(x) for x in args.map_id)}]"
        )
    )
    print(f"[train] Map     : {_map_line}")
    if args.max_days is not None:
        print(f"[train] Days    : max_days={int(args.max_days)} (engine calendar cap)")
    if args.tier or args.co_p0 is not None or args.co_p1 is not None:
        print(
            f"[train] Curriculum: tier={args.tier!r} co_p0={args.co_p0} co_p1={args.co_p1} "
            f"broad_prob={args.curriculum_broad_prob} tag={args.curriculum_tag!r}"
        )
    if args.live_games_id:
        print(
            f"[train] Live PPO: games_id={args.live_games_id} "
            f"seats={args.live_learner_seats!r} dir={args.live_snapshot_dir!r}"
        )

    _sync_worker_inherited_env_flags(args)

    if args.log_replay_frames:
        os.environ["AWBW_LOG_REPLAY_FRAMES"] = "1"

    if args.fps_diag:
        os.environ["AWBW_FPS_DIAG"] = "1"

    if args.machine_id is not None:
        mid = str(args.machine_id).strip()
        if mid:
            os.environ["AWBW_MACHINE_ID"] = mid

    # Non-interactive: if machine_id is in the environment (e.g. .env) but CLI did not
    # pass --fps-diag, still stamp fps_diag for throughput visibility in fleet layouts.
    if (os.environ.get("AWBW_MACHINE_ID") or "").strip() and not (
        (os.environ.get("AWBW_FPS_DIAG") or "").strip()
    ):
        os.environ["AWBW_FPS_DIAG"] = "1"

    # Defaults used by fleet launchers as well; set here for direct ``train.py``
    # runs before ``rl.self_play`` imports ``rl.env`` / ``engine.game``.
    os.environ.setdefault("AWBW_REWARD_SHAPING", "phi")
    os.environ.setdefault("AWBW_TIME_COST", "0.00005")
    os.environ.setdefault("AWBW_TRUNCATION_PENALTY", "0.25")

    _env_flags = [
        ("AWBW_TIME_COST", os.environ.get("AWBW_TIME_COST")),
        ("AWBW_TRUNCATION_PENALTY", os.environ.get("AWBW_TRUNCATION_PENALTY")),
        ("AWBW_INCOME_TERM_COEF", os.environ.get("AWBW_INCOME_TERM_COEF")),
        ("AWBW_BUILD_MASK_INFANTRY_ONLY", os.environ.get("AWBW_BUILD_MASK_INFANTRY_ONLY")),
        ("AWBW_LOG_REPLAY_FRAMES", os.environ.get("AWBW_LOG_REPLAY_FRAMES")),
        ("AWBW_FPS_DIAG", os.environ.get("AWBW_FPS_DIAG")),
        ("AWBW_LEARNER_GREEDY_MIX", os.environ.get("AWBW_LEARNER_GREEDY_MIX")),
        ("AWBW_EGOCENTRIC_EPISODE_PROB", os.environ.get("AWBW_EGOCENTRIC_EPISODE_PROB")),
        ("AWBW_CAPTURE_MOVE_GATE", os.environ.get("AWBW_CAPTURE_MOVE_GATE")),
        ("AWBW_SEAT_BALANCE", os.environ.get("AWBW_SEAT_BALANCE")),
        ("AWBW_LEARNER_SEAT", os.environ.get("AWBW_LEARNER_SEAT")),
        ("AWBW_PFSP", os.environ.get("AWBW_PFSP")),
        ("AWBW_ASYNC_VEC", os.environ.get("AWBW_ASYNC_VEC")),
        ("AWBW_REWARD_SHAPING", os.environ.get("AWBW_REWARD_SHAPING")),
        ("AWBW_PAIRWISE_ZERO_SUM_REWARD", os.environ.get("AWBW_PAIRWISE_ZERO_SUM_REWARD")),
        ("AWBW_TRACK_PER_WORKER_TIMES", os.environ.get("AWBW_TRACK_PER_WORKER_TIMES")),
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

    max_env_steps = (
        None if args.max_env_steps <= 0 else int(args.max_env_steps)
    )
    max_p1_microsteps = (
        None if args.max_p1_microsteps <= 0 else int(args.max_p1_microsteps)
    )

    max_turns = args.max_days
    if max_turns is not None:
        max_turns_i = int(max_turns)
        if max_turns_i < 1:
            parser.error("--max-days must be >= 1 when provided")
        max_turns = max_turns_i

    live_games_id = list(args.live_games_id) if args.live_games_id else None
    live_learner_seats = None
    if args.live_learner_seats:
        live_learner_seats = [
            int(x.strip()) for x in str(args.live_learner_seats).split(",") if x.strip()
        ]

    from rl.self_play import SelfPlayTrainer
    trainer = SelfPlayTrainer(
        total_timesteps=args.iters,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=device,
        save_every=args.save_every,
        publish_latest_each_save=bool(args.publish_latest_each_save),
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
        mcts_mode=args.mcts_mode,
        mcts_sims=args.mcts_sims,
        mcts_c_puct=args.mcts_c_puct,
        mcts_dirichlet_alpha=args.mcts_dirichlet_alpha,
        mcts_dirichlet_epsilon=args.mcts_dirichlet_epsilon,
        mcts_temperature=args.mcts_temperature,
        mcts_min_depth=args.mcts_min_depth,
        mcts_root_plans=args.mcts_root_plans,
        mcts_max_plan_actions=args.mcts_max_plan_actions,
        max_env_steps=max_env_steps,
        max_p1_microsteps=max_p1_microsteps,
        max_turns=max_turns,
        live_games_id=live_games_id,
        live_learner_seats=live_learner_seats,
        live_snapshot_dir=args.live_snapshot_dir,
        training_backend=args.training_backend,
        async_unroll_length=args.async_unroll_length,
        async_learner_batch=args.async_learner_batch,
        async_queue_max=args.async_queue_max,
        async_gpu_opponent_permits_subtract=args.async_gpu_opponent_permits_subtract,
        async_learner_forward_chunk=args.async_learner_forward_chunk,
        dual_gradient_self_play=bool(args.dual_gradient_self_play),
        dual_gradient_hist_prob=float(args.dual_gradient_hist_prob),
        opening_book_path=args.opening_book,
        opening_book_seat=getattr(args, "opening_book_seat", 1),
        opening_book_seats=getattr(args, "opening_book_seats", "both"),
        opening_book_prob=getattr(args, "opening_book_prob", 1.0),
        opening_book_strict_co=getattr(args, "opening_book_strict_co", False),
        opening_book_max_day=(
            None
            if int(getattr(args, "opening_book_days", 0) or 0) <= 0
            else int(args.opening_book_days)
        ),
        opening_book_seed=int(getattr(args, "opening_book_seed", 0) or 0),
        cop_disable_per_seat_p=getattr(args, "cop_disable_per_seat_p", None),
    )
    _install_sigint_first_only()
    trainer.train()


if __name__ == "__main__":
    # SubprocVecEnv on Windows requires the freeze_support guard
    import multiprocessing
    multiprocessing.freeze_support()
    main()
