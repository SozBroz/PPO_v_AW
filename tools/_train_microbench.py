"""One-off env microbench (random P0, random op, narrow map). Delete after use."""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import random
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
POOL = REPO / "data" / "gl_map_pool.json"


def _load_narrow_map_pool() -> list[dict[str, Any]]:
    with POOL.open(encoding="utf-8") as f:
        full: list[dict[str, Any]] = json.load(f)
    m123 = [m for m in full if m.get("map_id") == 123858]
    if not m123:
        raise SystemExit("map_id 123858 not found in gl_map_pool.json")
    return m123


def _legal_random_step(env: Any, rng: random.Random) -> bool:
    """One P0 env step with random legal action. Returns True if term or trunc (episode cut)."""
    mask = env.action_masks()
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return True
    a = int(rng.choice(legal))
    _obs, _r, term, trunc, _info = env.step(a)
    return bool(term or trunc)


def _run_unprofiled(
    env: Any,
    rng: random.Random,
    n_steps: int,
    reset_seed: int,
) -> tuple[float, int]:
    env.reset(seed=reset_seed)
    t0 = time.perf_counter()
    completed = 0
    for _ in range(n_steps):
        cut = _legal_random_step(env, rng)
        completed += 1
        if cut:
            env.reset(seed=rng.randrange(2**31))
    return time.perf_counter() - t0, completed


def _run_profiled(
    env: Any,
    rng: random.Random,
    n_steps: int,
    reset_seed: int,
    pr: cProfile.Profile,
) -> int:
    env.reset(seed=reset_seed)
    pr.enable()
    try:
        for _ in range(n_steps):
            cut = _legal_random_step(env, rng)
            if cut:
                env.reset(seed=rng.randrange(2**31))
    finally:
        pr.disable()
    return n_steps


def _collect_target_rows(pr: cProfile.Profile) -> list[dict[str, Any]]:
    st = pstats.Stats(pr)
    st.strip_dirs()
    out: list[dict[str, Any]] = []
    targets = ("get_legal_actions", "encode_state", "_engine_step_with_belief")
    for (fn, line, name), (cc, nc, tt, ct, _cs) in st.stats.items():
        for t in targets:
            if t in name or name == t:
                out.append(
                    {
                        "file": str(fn),
                        "line": line,
                        "name": name,
                        "n_calls": int(nc),
                        "cumtime_s": float(ct),
                        "tottime_s": float(tt),
                    }
                )
                break
    out.sort(key=lambda r: r["cumtime_s"], reverse=True)
    return out


def _top_cumulative_report(pr: cProfile.Profile, n: int) -> str:
    buf = StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.strip_dirs()
    st.sort_stats("cumulative")
    st.print_stats(n)
    return buf.getvalue()


def main() -> None:
    os.environ["AWBW_LOG_REPLAY_FRAMES"] = "0"
    from rl.env import AWBWEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000, help="P0 steps per leg (default 2000)")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG / first reset seed")
    ap.add_argument(
        "--narrow",
        action="store_true",
        default=True,
        help="Use map 123858, T3, co 1,1 (default: on)",
    )
    ap.add_argument("--no-narrow", action="store_false", dest="narrow")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write machine-readable summary to this file",
    )
    args = ap.parse_args()
    n_steps: int = max(1, int(args.steps))
    seed: int = int(args.seed)

    if args.narrow:
        mpool = _load_narrow_map_pool()
        env = AWBWEnv(
            map_pool=mpool,
            co_p0=1,
            co_p1=1,
            tier_name="T3",
            curriculum_broad_prob=0.0,
            opponent_policy=None,
        )
    else:
        env = AWBWEnv(
            curriculum_broad_prob=1.0,
            opponent_policy=None,
        )

    results: dict[str, Any] = {
        "steps_per_leg": n_steps,
        "seed": seed,
        "narrow": bool(args.narrow),
    }

    # Two unprofiled runs (repeatability)
    times: list[tuple[str, float, float]] = []
    for i in (0, 1):
        r = random.Random(seed + 1000 * i)
        wall, done_steps = _run_unprofiled(env, r, n_steps, reset_seed=seed + i * 17)
        sps = done_steps / wall if wall > 0 else 0.0
        name = f"unprofiled_run_{i+1}"
        print(f"{name} wall_s={wall:.4f} env_steps_s={sps:.1f}")
        times.append((name, wall, sps))
    mwall = (times[0][1] + times[1][1]) / 2.0
    msps = (times[0][2] + times[1][2]) / 2.0
    rel = abs(times[0][2] - times[1][2]) / max(msps, 1e-9) * 100.0
    print(f"mean env_steps_s={msps:.1f}  pair rel_diff%={rel:.1f}")
    results["unprofiled"] = {
        "runs": [
            {"name": t[0], "wall_s": t[1], "env_steps_s": t[2]} for t in times
        ],
        "mean_env_steps_s": msps,
        "rel_percent_diff_vs_pair_mean": rel,
    }

    # One profiled run (full leg)
    pr = cProfile.Profile()
    r3 = random.Random(seed + 3000)
    t_prof0 = time.perf_counter()
    _run_profiled(env, r3, n_steps, seed + 99, pr)
    t_prof1 = time.perf_counter()
    print(f"profiled_leg wall_s={t_prof1 - t_prof0:.4f} (includes profiler overhead)")

    top_txt = _top_cumulative_report(pr, 15)
    print()
    print("--- Top 15 by cumulative (cProfile) ---")
    print(top_txt)
    target_rows = _collect_target_rows(pr)
    print("--- Filtered: get_legal_actions / encode_state / _engine_step_with_belief ---")
    for row in target_rows[:40]:
        print(
            f"  {row['cumtime_s']:.3f}s cum  {row['n_calls']} calls  {row['name']}"
        )

    results["profiled"] = {
        "top_15_cumulative_text": top_txt,
        "target_function_rows": target_rows,
        "wall_s": t_prof1 - t_prof0,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print()
        print(f"Wrote {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
