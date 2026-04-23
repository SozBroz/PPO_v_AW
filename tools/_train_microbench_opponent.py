"""Microbench: policy opponent loop with configurable needs_observation (Phase 1a path)."""
from __future__ import annotations

import argparse
import cProfile
import json
import os
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


class _StubOpponent:
    """Configurable opponent for the microbench."""

    def __init__(self, *, declares_no_obs: bool) -> None:
        self._declares_no_obs = declares_no_obs
        self._env: Any = None

    def needs_observation(self) -> bool:
        return not self._declares_no_obs

    def attach_env(self, env: Any) -> None:
        self._env = env

    def __call__(self, obs: Any, mask: np.ndarray) -> int:
        idx = int(np.flatnonzero(mask)[0]) if mask.any() else 0
        return idx


def _load_narrow_map_pool() -> list[dict[str, Any]]:
    with POOL.open(encoding="utf-8") as f:
        full: list[dict[str, Any]] = json.load(f)
    m123 = [m for m in full if m.get("map_id") == 123858]
    if not m123:
        raise SystemExit("map_id 123858 not found in gl_map_pool.json")
    return m123


def _legal_random_step(env: Any, rng: random.Random) -> bool:
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
) -> None:
    env.reset(seed=reset_seed)
    pr.enable()
    try:
        for _ in range(n_steps):
            cut = _legal_random_step(env, rng)
            if cut:
                env.reset(seed=rng.randrange(2**31))
    finally:
        pr.disable()


def _top_cumulative_rows(pr: cProfile.Profile, n: int) -> list[dict[str, Any]]:
    import pstats

    rows: list[dict[str, Any]] = []
    for (fn, line, name), (_cc, nc, tt, ct, _cs) in pstats.Stats(pr).stats.items():
        rows.append(
            {
                "file": str(fn),
                "line": line,
                "name": name,
                "n_calls": int(nc),
                "cumtime_s": float(ct),
                "tottime_s": float(tt),
            }
        )
    rows.sort(key=lambda r: r["cumtime_s"], reverse=True)
    return rows[:n]


def _top_cumulative_report(pr: cProfile.Profile, n: int) -> str:
    import pstats

    buf = StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.strip_dirs()
    st.sort_stats("cumulative")
    st.print_stats(n)
    return buf.getvalue()


def _cumtime_by_exact_name(pr: cProfile.Profile, target: str) -> dict[str, Any] | None:
    import pstats

    best: dict[str, Any] | None = None
    for (fn, line, name), (_cc, nc, tt, ct, _cs) in pstats.Stats(pr).stats.items():
        if name == target:
            row = {
                "file": str(fn),
                "line": line,
                "name": name,
                "n_calls": int(nc),
                "cumtime_s": float(ct),
                "tottime_s": float(tt),
            }
            if best is None or row["cumtime_s"] > best["cumtime_s"]:
                best = row
    return best


def _make_env(declares_no_obs: bool) -> Any:
    from rl.env import AWBWEnv

    mpool = _load_narrow_map_pool()
    stub = _StubOpponent(declares_no_obs=declares_no_obs)
    env = AWBWEnv(
        map_pool=mpool,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
        curriculum_broad_prob=0.0,
        opponent_policy=stub,
    )
    stub.attach_env(env)
    return env


def main() -> None:
    os.environ["AWBW_LOG_REPLAY_FRAMES"] = "0"

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("logs/microbench_opponent_loop.json"),
    )
    args = ap.parse_args()
    n_steps = max(1, int(args.steps))
    seed = int(args.seed)

    # Pass A: needs_observation True (declares_no_obs=False)
    # Pass B: needs_observation False (declares_no_obs=True)
    results: dict[str, Any] = {
        "steps": n_steps,
        "seed": seed,
        "pass_a": {"label": "needs_observation=True (full P1 encode)", "declares_no_obs": False},
        "pass_b": {"label": "needs_observation=False (Phase 1a skip)", "declares_no_obs": True},
    }

    rng_a = random.Random(seed + 1000)
    env_a = _make_env(declares_no_obs=False)
    wall_a, done_a = _run_unprofiled(env_a, rng_a, n_steps, reset_seed=seed)
    sps_a = done_a / wall_a if wall_a > 0 else 0.0
    print(f"Pass A (full P1 obs) wall_s={wall_a:.4f} env_steps_s={sps_a:.1f}")

    rng_b = random.Random(seed + 1000)
    env_b = _make_env(declares_no_obs=True)
    wall_b, done_b = _run_unprofiled(env_b, rng_b, n_steps, reset_seed=seed)
    sps_b = done_b / wall_b if wall_b > 0 else 0.0
    print(f"Pass B (skip P1 obs)  wall_s={wall_b:.4f} env_steps_s={sps_b:.1f}")

    speedup_pct = (sps_b - sps_a) / sps_a * 100.0 if sps_a > 0 else 0.0
    wall_delta_s = wall_a - wall_b
    print(f"Speedup B vs A: {speedup_pct:+.1f}% env_steps/s  (wall delta {wall_delta_s:+.4f}s)")

    pr_a = cProfile.Profile()
    rng_pa = random.Random(seed + 3000)
    env_pa = _make_env(declares_no_obs=False)
    t0 = time.perf_counter()
    _run_profiled(env_pa, rng_pa, n_steps, seed + 99, pr_a)
    prof_wall_a = time.perf_counter() - t0

    pr_b = cProfile.Profile()
    rng_pb = random.Random(seed + 3000)
    env_pb = _make_env(declares_no_obs=True)
    t1 = time.perf_counter()
    _run_profiled(env_pb, rng_pb, n_steps, seed + 99, pr_b)
    prof_wall_b = time.perf_counter() - t1

    top_a = _top_cumulative_rows(pr_a, 10)
    top_b = _top_cumulative_rows(pr_b, 10)
    enc_a = _cumtime_by_exact_name(pr_a, "encode_state")
    enc_b = _cumtime_by_exact_name(pr_b, "encode_state")
    obs_a = _cumtime_by_exact_name(pr_a, "_get_obs")
    obs_b = _cumtime_by_exact_name(pr_b, "_get_obs")

    results["pass_a"]["unprofiled"] = {"wall_s": wall_a, "env_steps_s": sps_a, "completed_steps": done_a}
    results["pass_b"]["unprofiled"] = {"wall_s": wall_b, "env_steps_s": sps_b, "completed_steps": done_b}
    results["comparison"] = {
        "speedup_env_steps_s_percent": speedup_pct,
        "wall_delta_s_A_minus_B": wall_delta_s,
        "smoke_gate_min_speedup_pct": 5.0,
        "smoke_gate_passed": bool(speedup_pct >= 5.0),
    }
    results["pass_a"]["profiled_wall_s"] = prof_wall_a
    results["pass_a"]["top_10_cumulative"] = top_a
    results["pass_a"]["encode_state"] = enc_a
    results["pass_a"]["_get_obs"] = obs_a
    results["pass_a"]["top_10_cumulative_text"] = _top_cumulative_report(pr_a, 10)

    results["pass_b"]["profiled_wall_s"] = prof_wall_b
    results["pass_b"]["top_10_cumulative"] = top_b
    results["pass_b"]["encode_state"] = enc_b
    results["pass_b"]["_get_obs"] = obs_b
    results["pass_b"]["top_10_cumulative_text"] = _top_cumulative_report(pr_b, 10)

    if enc_a and enc_b:
        results["comparison"]["encode_state_cumtime_delta_A_minus_B_s"] = (
            enc_a["cumtime_s"] - enc_b["cumtime_s"]
        )
    if obs_a and obs_b:
        results["comparison"]["_get_obs_cumtime_delta_A_minus_B_s"] = (
            obs_a["cumtime_s"] - obs_b["cumtime_s"]
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("--- Pass A: top 10 cumulative (cProfile) ---")
    print(results["pass_a"]["top_10_cumulative_text"])
    print("--- Pass B: top 10 cumulative (cProfile) ---")
    print(results["pass_b"]["top_10_cumulative_text"])

    if not results["comparison"]["smoke_gate_passed"]:
        print()
        print("SMOKE GATE FAILED: Pass B < 5% faster than Pass A — inspect Phase 1a wiring.")

    print()
    print(f"Wrote {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
