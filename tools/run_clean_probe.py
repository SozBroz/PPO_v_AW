#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a clean probe: short eval with all scaffolds forced to zero.

A clean probe verifies that the policy can function without opening books,
capture gates, or teacher overrides. It is run periodically during stub and
decay sub-stages to detect hidden scaffold dependency before advancing.

Usage::

    python tools/run_clean_probe.py \\
        --checkpoint checkpoints/latest.zip \\
        --out results/clean_probe_d0_stub.json \\
        --games 64 \\
        --map-pool data/gl_map_pool.json \\
        --seed 42

Output is consumed by :func:`tools.curriculum_advisor.compute_proposal`,
which reads the clean probe results from the output file and uses them in
sub-stage transition decisions.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path or URL to the model checkpoint zip to evaluate.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path for clean probe results.",
    )
    ap.add_argument(
        "--games",
        type=int,
        default=64,
        help="Number of games per clean probe (default 64).",
    )
    ap.add_argument(
        "--map-pool",
        type=str,
        default=None,
        help="JSON file listing allowed map IDs (default: all GL Std maps).",
    )
    ap.add_argument(
        "--co-pool",
        type=str,
        default=None,
        help="Comma-separated CO IDs (default: all enabled COs).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible game sampling.",
    )
    ap.add_argument(
        "--train-script",
        type=str,
        default="python train.py",
        help="Train.py invocation prefix.",
    )
    ap.add_argument(
        "--eval-script",
        type=str,
        default=None,
        help="Alternate eval script (default: train.py --eval-only).",
    )
    ap.add_argument(
        "--max-env-steps",
        type=int,
        default=4000,
        help="Max env steps per game (default 4000).",
    )
    ap.add_argument(
        "--timeout-per-game",
        type=float,
        default=120.0,
        help="Timeout per game in seconds (default 120).",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel workers (default 4).",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag identifying this probe's stage (e.g. 'stage_d0_gl_std_map_pool_stub').",
    )
    return ap


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CleanProbeResult:
    tag: str | None
    checkpoint: str
    seed: int
    games_run: int
    games_finished: int
    terminal_rate: float
    truncation_rate: float
    win_rate: float
    capture_sanity_rate: float        # fraction of games with captures_by_day5 >= 3
    mean_first_capture_step_p50: float
    mean_income_by_day5: float
    wall_time_s: float
    timestamp: str

    # Per-game raw data (kept small: just the fields needed for aggregation)
    first_capture_steps: list[float]
    captures_by_day5: list[int]
    incomes_by_day5: list[int]
    win_list: list[int]   # 1 = learner win, 0 = loss
    done_list: list[int]  # 1 = terminal, 0 = truncated

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["timestamp"] = self.timestamp
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────


def _sample_matchups(
    *,
    num_games: int,
    map_pool: list[int],
    co_pool: list[int],
    seed: int,
) -> list[tuple[int, int, int]]:
    """Sample (map_id, co_p0, co_p1) tuples for each game."""
    rng = random.Random(seed)
    matchups: list[tuple[int, int, int]] = []
    for _ in range(num_games):
        m = rng.choice(map_pool)
        c0 = rng.choice(co_pool)
        c1 = rng.choice(co_pool)
        matchups.append((m, c0, c1))
    return matchups


def run_single_game(
    *,
    checkpoint: str,
    map_id: int,
    co_p0: int,
    co_p1: int,
    seed: int,
    max_env_steps: int,
    timeout: float,
    train_cmd: str,
) -> dict:
    """Run one game and return a dict of parsed results.

    The game is run with all scaffolds forced to zero:
        --learner-greedy-mix 0
        (capture-move-gate omitted; default off)
        --opening-book-prob 0.0
        --cold-opponent random
    """
    env = {
        **dict(__import__("os").environ),
        "AWBW_SEED": str(seed),
    }
    cmd = [
        *train_cmd.split(),
        "--load", checkpoint,
        "--map-id", str(map_id),
        "--co-p0", str(co_p0),
        "--co-p1", str(co_p1),
        "--learner-greedy-mix", "0.0",
        "--opening-book-prob", "0.0",
        "--cold-opponent", "random",
        "--n-envs", "1",
        "--max-env-steps", str(max_env_steps),
        "--tier", "T4",
        # Force no broad sampling for clean probe determinism
        "--curriculum-broad-prob", "0.0",
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.monotonic() - started
        # Try to parse game result from stdout/stderr
        # The train.py eval path logs JSON lines to stdout
        result = _parse_game_output(proc.stdout + proc.stderr, elapsed)
        result["map_id"] = map_id
        result["co_p0"] = co_p0
        result["co_p1"] = co_p1
        result["seed"] = seed
        return result
    except subprocess.TimeoutExpired:
        return {
            "map_id": map_id, "co_p0": co_p0, "co_p1": co_p1,
            "seed": seed, "done": False, "truncated": True,
            "win": False, "turns": 0, "days": 0, "wall_s": timeout,
            "first_learner_capture_step": None, "captures_by_day5": 0,
            "income_by_day5": 0, "error": "timeout",
        }
    except Exception as exc:
        return {
            "map_id": map_id, "co_p0": co_p0, "co_p1": co_p1,
            "seed": seed, "done": False, "truncated": True,
            "win": False, "turns": 0, "days": 0, "wall_s": 0.0,
            "first_learner_capture_step": None, "captures_by_day5": 0,
            "income_by_day5": 0, "error": str(exc),
        }


def _parse_game_output(output: str, elapsed: float) -> dict:
    """Try to extract game result fields from train.py output."""
    # Look for JSON lines in output
    result = {
        "done": False, "truncated": False, "win": False,
        "turns": 0, "days": 0, "wall_s": elapsed,
        "first_learner_capture_step": None,
        "captures_by_day5": 0,
        "income_by_day5": 0,
    }
    for line in output.splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "event" in obj:
                if obj["event"] == "game_done" or obj.get("done"):
                    result["done"] = True
                    result["win"] = bool(obj.get("learner_win", obj.get("winner") == 0))
                    _td = obj.get("days", obj.get("turns", 0))
                    result["turns"] = int(_td)
                    result["days"] = int(_td)
                    result["truncated"] = bool(obj.get("truncated", False))
                    result["first_learner_capture_step"] = obj.get("first_p0_capture_p0_step")
                    result["captures_by_day5"] = int(obj.get("captures_completed_p0", 0))
                    result["income_by_day5"] = int(obj.get("income_by_day5_p0", 0))
                    break
        except json.JSONDecodeError:
            continue
    return result


def aggregate_results(games: list[dict]) -> CleanProbeResult:
    """Aggregate per-game results into summary statistics."""
    finished = [g for g in games if g.get("done") or not g.get("error")]
    n = len(finished) or 1

    first_caps: list[float] = []
    caps_by_d5: list[int] = []
    incomes_d5: list[int] = []
    wins: list[int] = []
    done_list: list[int] = []

    for g in finished:
        fc = g.get("first_learner_capture_step")
        first_caps.append(float(fc) if fc is not None else 999.0)
        caps_by_d5.append(int(g.get("captures_by_day5", 0)))
        incomes_d5.append(int(g.get("income_by_day5", 0)))
        wins.append(1 if g.get("win") else 0)
        done_list.append(1 if g.get("done") else 0)

    cap_sanity = sum(1 for c in caps_by_d5 if c >= 3) / max(1, len(caps_by_d5))
    terminal_rate = sum(done_list) / max(1, len(done_list))
    truncation_rate = sum(1 for g in finished if g.get("truncated")) / max(1, len(finished))
    win_rate = sum(wins) / max(1, len(wins))
    mean_first_cap = sum(first_caps) / len(first_caps) if first_caps else 999.0
    mean_income_d5 = sum(incomes_d5) / max(1, len(incomes_d5))

    wall = max(g.get("wall_s", 0) for g in games) if games else 0.0

    return CleanProbeResult(
        tag=None,
        checkpoint="",
        seed=0,
        games_run=len(games),
        games_finished=len(finished),
        terminal_rate=terminal_rate,
        truncation_rate=truncation_rate,
        win_rate=win_rate,
        capture_sanity_rate=cap_sanity,
        mean_first_capture_step_p50=mean_first_cap,
        mean_income_by_day5=mean_income_d5,
        wall_time_s=wall,
        timestamp=datetime.now(timezone.utc).isoformat(),
        first_capture_steps=first_caps,
        captures_by_day5=caps_by_d5,
        incomes_by_day5=incomes_d5,
        win_list=wins,
        done_list=done_list,
    )


def run_clean_probe(args: argparse.Namespace) -> CleanProbeResult:
    """Run the full clean probe."""
    rng = random.Random(args.seed)

    # Determine map pool
    if args.map_pool:
        mp_path = Path(args.map_pool)
        if mp_path.is_file():
            mp_data = json.loads(mp_path.read_text(encoding="utf-8"))
            map_pool = [int(m["map_id"]) for m in mp_data.get("maps", [])]
        else:
            map_pool = [int(x) for x in mp_data.get("maps", []) if x]
    else:
        # Default: try to load GL map pool
        repo_root = Path(__file__).resolve().parents[1]
        gl_pool_path = repo_root / "data" / "gl_map_pool.json"
        if gl_pool_path.is_file():
            mp_data = json.loads(gl_pool_path.read_text(encoding="utf-8"))
            map_pool = [int(m["map_id"]) for m in mp_data.get("maps", [])]
        else:
            map_pool = [123858]  # fallback to Misery

    # Determine CO pool
    if args.co_pool:
        co_pool = [int(c.strip()) for c in args.co_pool.split(",")]
    else:
        co_pool = [14, 0, 1, 2, 3]  # Jess, Andy, Max, Sami, Eagle

    matchups = _sample_matchups(
        num_games=args.games,
        map_pool=map_pool,
        co_pool=co_pool,
        seed=args.seed,
    )

    train_cmd = args.eval_script if args.eval_script else (args.train_script + " --eval-only")

    print(f"[clean_probe] Running {args.games} games with scaffolds FORCED TO ZERO")
    print(f"[clean_probe] Maps: {len(map_pool)}, COs: {co_pool}")
    print(f"[clean_probe] Checkpoint: {args.checkpoint}")

    games: list[dict] = []
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = []
        for i, (m, c0, c1) in enumerate(matchups):
            game_seed = rng.randint(0, 2**31 - 1)
            f = ex.submit(
                run_single_game,
                checkpoint=args.checkpoint,
                map_id=m, co_p0=c0, co_p1=c1,
                seed=game_seed,
                max_env_steps=args.max_env_steps,
                timeout=args.timeout_per_game,
                train_cmd=train_cmd,
            )
            futures.append(f)
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            g = f.result()
            games.append(g)
            if (i + 1) % 10 == 0:
                print(f"[clean_probe] {i+1}/{args.games} games done")

    result = aggregate_results(games)
    result.tag = args.tag
    result.checkpoint = args.checkpoint
    result.seed = args.seed
    return result


def write_results(result: CleanProbeResult, out_path: Path) -> None:
    """Write results to JSON, creating parent directories as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), indent=2)
    out_path.write_text(payload, encoding="utf-8")
    print(f"[clean_probe] Results written to {out_path}")


def main() -> int:
    ap = _build_arg_parser()
    args = ap.parse_args()
    result = run_clean_probe(args)
    write_results(result, args.out)
    print(f"[clean_probe] Summary: terminal={result.terminal_rate:.3f}  "
          f"truncation={result.truncation_rate:.3f}  win_rate={result.win_rate:.3f}  "
          f"cap_sanity={result.capture_sanity_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
