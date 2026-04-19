"""
Estimate disk/stdout logging overhead vs PPO work for AWBW self-play training.

Encodes the quantitative model from the logging-vs-training-pace analysis:
default hyperparameters match rl/self_play.py (MaskablePPO) and train.py defaults.

Usage:
  python -m analysis.logging_overhead_estimate
  python -m analysis.logging_overhead_estimate --mean-episode-steps 1500 --chunk-wall-sec 420
"""

from __future__ import annotations

import argparse


def rollouts_per_chunk(total_timesteps: int, n_steps: int, n_envs: int) -> float:
    steps_per_rollout = n_steps * n_envs
    return total_timesteps / steps_per_rollout


def completions_per_chunk(save_every: int, mean_episode_steps: float) -> float:
    """Approximate episode completions per outer chunk (all envs)."""
    if mean_episode_steps <= 0:
        return float("nan")
    return save_every / mean_episode_steps


def main() -> None:
    p = argparse.ArgumentParser(
        description="Estimate logging vs training time (order-of-magnitude)."
    )
    p.add_argument(
        "--n-envs", type=int, default=6, help="Parallel envs (default: 6, train.py)"
    )
    p.add_argument(
        "--n-steps", type=int, default=512, help="PPO n_steps per env (default: 512)"
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=50_000,
        help="Timesteps per model.learn chunk (default: 50_000)",
    )
    p.add_argument(
        "--mean-episode-steps",
        type=float,
        default=1_500.0,
        help="Mean P0 env-steps per completed episode (tune to your runs)",
    )
    p.add_argument(
        "--ms-per-game-log",
        type=float,
        default=2.0,
        help="Assumed ms per SQLite+JSONL append per episode (SSD order-of-magnitude)",
    )
    p.add_argument(
        "--pessimistic-ms-per-game-log",
        type=float,
        default=5.0,
        help="Pessimistic ms per game log for upper-bound row",
    )
    p.add_argument(
        "--checkpoint-mb",
        type=float,
        default=40.0,
        help="Assumed zip size per model.save (MB)",
    )
    p.add_argument(
        "--checkpoint-writes",
        type=int,
        default=2,
        help="model.save calls per chunk (default: 2, numbered + latest)",
    )
    p.add_argument(
        "--disk-mb-per-sec",
        type=float,
        default=200.0,
        help="Assumed sequential write throughput (MB/s) for checkpoint math",
    )
    p.add_argument(
        "--chunk-wall-sec",
        type=float,
        default=300.0,
        help="Example wall-clock seconds for one save-every chunk (env+GPU)",
    )
    p.add_argument(
        "--stdout-ms-per-chunk",
        type=float,
        default=1.0,
        help="Order-of-magnitude ms for SB3 verbose + self_play prints per chunk",
    )
    args = p.parse_args()

    n_steps, n_envs = args.n_steps, args.n_envs
    save_every = args.save_every
    L = args.mean_episode_steps

    steps_per_rollout = n_steps * n_envs
    r = rollouts_per_chunk(save_every, n_steps, n_envs)
    n_complete = completions_per_chunk(save_every, L)

    game_log_sec_typical = (n_complete * args.ms_per_game_log) / 1000.0
    game_log_sec_pess = (n_complete * args.pessimistic_ms_per_game_log) / 1000.0

    ckpt_bytes = args.checkpoint_mb * 1_000_000 * args.checkpoint_writes
    ckpt_sec = (ckpt_bytes / 1_000_000) / args.disk_mb_per_sec if args.disk_mb_per_sec > 0 else float("inf")

    stdout_sec = args.stdout_ms_per_chunk / 1000.0
    wall = args.chunk_wall_sec

    def pct(t: float) -> float:
        return 100.0 * t / wall if wall > 0 else float("inf")

    print("AWBW training - logging vs PPO (order-of-magnitude)\n")
    print(f"  n_steps={n_steps}  n_envs={n_envs}  -> steps/rollout = {steps_per_rollout:,}")
    print(f"  save_every (chunk) = {save_every:,} env steps")
    print(f"  rollouts per chunk ~ {r:.2f}")
    print(f"  assumed mean P0 steps/episode L = {L:g}")
    print(f"  completions per chunk ~ save_every/L = {n_complete:.1f}\n")

    print("Estimated time per chunk (seconds):")
    print(f"  game_log.jsonl (typical, {args.ms_per_game_log:g} ms/episode): {game_log_sec_typical:.3f}  (~{pct(game_log_sec_typical):.3f}% of chunk wall {wall:g}s)")
    print(
        f"  game_log.jsonl (pessimistic, {args.pessimistic_ms_per_game_log:g} ms/episode): "
        f"{game_log_sec_pess:.3f}  (~{pct(game_log_sec_pess):.3f}%)"
    )
    print(
        f"  checkpoints ({args.checkpoint_writes}x ~{args.checkpoint_mb:g} MB @ {args.disk_mb_per_sec:g} MB/s): "
        f"{ckpt_sec:.3f}  (~{pct(ckpt_sec):.3f}%)"
    )
    print(f"  stdout (SB3 verbose + prints, order-of-magnitude): {stdout_sec:.4f}  (~{pct(stdout_sec):.4f}%)\n")

    print("Verdict (same as analysis plan): PPO env + GPU dominates; ordinary JSONL/stdout")
    print("is usually sub-percent of chunk wall time unless disk is pathological or")
    print("AWBW_LOG_REPLAY_FRAMES=1 (large frames) is enabled.\n")

    print("Completions vs L (save_every={:,}):".format(save_every))
    for example_l in (500, 1_500, 3_000):
        c = completions_per_chunk(save_every, float(example_l))
        print(f"  L = {example_l:5d}  ->  ~{c:6.1f} completions/chunk")


if __name__ == "__main__":
    main()
