#!/usr/bin/env python3
"""Bo3 playoff: challenger zip = P0, defender zip = P1; first to 2 wins; replace defender if challenger wins.

With ``--parallel`` (default), runs up to three games in **separate processes**
(fresh model loads per worker — safe on Windows). Otherwise games run sequentially.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.ai_vs_ai import _MaxCalendarDaysAction


class _FixedZipOpponent:
    def __init__(self, zip_path: Path, *, deterministic: bool = False) -> None:
        self._path = str(zip_path.resolve())
        self._model = None
        self._det = bool(deterministic)

    def __call__(self, obs, mask):
        if self._model is None:
            from rl.ckpt_compat import load_maskable_ppo_compat

            self._model = load_maskable_ppo_compat(self._path, device="cpu", cache=True)
        act, _ = self._model.predict(obs, action_masks=mask, deterministic=self._det)
        return int(act)


def _play_one_game(env, p0_model, *, rng_seed: int, deterministic: bool) -> int:
    obs, _ = env.reset(seed=rng_seed)
    done = False
    while not done:
        mask = env.action_masks()
        act, _ = p0_model.predict(
            obs, action_masks=mask, deterministic=bool(deterministic)
        )
        obs, _r, term, trunc, _ = env.step(int(act))
        done = bool(term or trunc)
    w = env.unwrapped.state.winner
    return -1 if w is None else int(w)


def _worker_bo3_game(payload: dict) -> tuple[int, int]:
    """Picklable worker: return (game_index, winner)."""
    from sb3_contrib.common.wrappers import ActionMasker

    from rl.ckpt_compat import load_maskable_ppo_compat
    from rl.env import AWBWEnv

    game_i = int(payload["game_i"])
    seed = int(payload["seed"])
    ch = Path(payload["challenger"])
    df = Path(payload["defender"])
    map_pool = json.loads(payload["map_pool_json"])
    tier = payload["tier"]
    co_p0 = payload["co_p0"]
    co_p1 = payload["co_p1"]
    deterministic = bool(payload["deterministic"])
    max_env_steps = payload.get("max_env_steps")
    max_p1_microsteps = payload.get("max_p1_microsteps")
    max_turns = payload.get("max_turns")
    if max_turns is None:
        max_turns = payload.get("max_days")
        def __init__(self) -> None:
            self._m = None

        def __call__(self, obs, mask):
            if self._m is None:
                self._m = load_maskable_ppo_compat(str(df), device="cpu", cache=True)
            a, _ = self._m.predict(obs, action_masks=mask, deterministic=deterministic)
            return int(a)

    opp = _Opp()
    env_kw: dict = dict(
        map_pool=map_pool,
        opponent_policy=opp,
        co_p0=co_p0,
        co_p1=co_p1,
        tier_name=tier,
        curriculum_tag="bo3_playoff",
    )
    if max_env_steps is not None:
        env_kw["max_env_steps"] = int(max_env_steps)
    if max_p1_microsteps is not None:
        env_kw["max_p1_microsteps"] = int(max_p1_microsteps)
    if max_turns is not None:
        env_kw["max_turns"] = int(max_turns)
    env = ActionMasker(
        AWBWEnv(**env_kw),
        lambda e: e.action_masks(),
    )
    p0 = load_maskable_ppo_compat(str(ch), device="cpu", cache=True)
    w = _play_one_game(env, p0, rng_seed=seed, deterministic=deterministic)
    return game_i, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", type=Path, required=True)
    ap.add_argument("--defender", type=Path, default=Path("checkpoints/latest.zip"))
    ap.add_argument("--first-to", type=int, default=2)
    ap.add_argument("--max-games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-id", type=int, default=None)
    ap.add_argument("--tier", type=str, default=None)
    ap.add_argument("--co-p0", type=int, default=None)
    ap.add_argument("--co-p1", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run up to three games in parallel processes (default: true). Use --no-parallel for sequential.",
    )
    ap.add_argument(
        "--max-env-steps",
        type=int,
        default=100,
        help="Max P0 env.step calls per game (0 = unlimited). Caps playoff length.",
    )
    ap.add_argument(
        "--max-p1-microsteps",
        type=int,
        default=None,
        help="Override cap on P1 engine steps per P0 step (default: derived from max-env-steps).",
    )
    ap.add_argument(
        "--max-days",
        "--max-turns",
        dest="max_days",
        type=int,
        default=None,
        metavar="N",
        action=_MaxCalendarDaysAction,
        help="End-inclusive engine calendar day cap (alias --max-turns deprecated).",
    )
    args = ap.parse_args()
    max_env_steps = (
        None if args.max_env_steps is None or args.max_env_steps <= 0 else int(args.max_env_steps)
    )
    if args.max_days is not None and int(args.max_days) < 1:
        raise SystemExit("--max-days must be >= 1")
    ch, df = args.challenger.resolve(), args.defender.resolve()
    if not ch.is_file():
        raise SystemExit(f"Missing challenger: {ch}")
    if not df.is_file():
        raise SystemExit(f"Missing defender: {df}")

    from sb3_contrib.common.wrappers import ActionMasker

    from rl.ckpt_compat import (
        clear_maskable_ppo_load_cache,
        delete_eval_snapshots,
        load_maskable_ppo_compat,
        snapshot_eval_checkpoints,
    )
    from rl.env import AWBWEnv
    from rl.self_play import POOL_PATH

    import numpy as np

    snap_paths: tuple[Path, ...] | None = None
    try:
        run_id, (ch_snap, df_snap) = snapshot_eval_checkpoints(
            [("challenger", ch), ("defender", df)]
        )
        print(f"[bo3] frozen snapshots run_id={run_id}")
        print(f"[bo3]   challenger -> {ch_snap}")
        print(f"[bo3]   defender -> {df_snap}")
        snap_paths = (ch_snap, df_snap)

        with Path(POOL_PATH).open(encoding="utf-8") as f:
            map_pool = json.load(f)
        if args.map_id is not None:
            map_pool = [m for m in map_pool if m.get("map_id") == args.map_id]
            if not map_pool:
                raise SystemExit("no maps")

        pool_json = json.dumps(map_pool)
        rng = np.random.default_rng(args.seed)
        cw = dw = games = 0

        if args.parallel:
            batch = min(3, args.max_games)
            seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(batch)]
            payloads = [
                {
                    "game_i": i + 1,
                    "seed": seeds[i],
                    "challenger": str(ch_snap),
                    "defender": str(df_snap),
                    "map_pool_json": pool_json,
                    "tier": args.tier,
                    "co_p0": args.co_p0,
                    "co_p1": args.co_p1,
                    "deterministic": args.deterministic,
                    "max_env_steps": max_env_steps,
                    "max_p1_microsteps": args.max_p1_microsteps,
                    "max_turns": args.max_days,
                    "max_days": args.max_days,
                }
                for i in range(batch)
            ]
            with ProcessPoolExecutor(max_workers=batch) as ex:
                futs = {ex.submit(_worker_bo3_game, p): p for p in payloads}
                for fut in as_completed(futs):
                    gi, w = fut.result()
                    print(f"[bo3] game {gi} winner={w}")
                    games += 1
                    if w == 0:
                        cw += 1
                    elif w == 1:
                        dw += 1
            while cw < args.first_to and dw < args.first_to and games < args.max_games:
                games += 1
                seed = int(rng.integers(0, 2**31 - 1))
                w = _worker_bo3_game(
                    {
                        "game_i": games,
                        "seed": seed,
                        "challenger": str(ch_snap),
                        "defender": str(df_snap),
                        "map_pool_json": pool_json,
                        "tier": args.tier,
                        "co_p0": args.co_p0,
                        "co_p1": args.co_p1,
                        "deterministic": args.deterministic,
                        "max_env_steps": max_env_steps,
                        "max_p1_microsteps": args.max_p1_microsteps,
                        "max_turns": args.max_days,
                        "max_days": args.max_days,
                    }
                )[1]
                print(f"[bo3] game {games} (sequential tail) winner={w}")
                if w == 0:
                    cw += 1
                elif w == 1:
                    dw += 1
        else:
            opp = _FixedZipOpponent(df_snap, deterministic=args.deterministic)
            env_kw: dict = dict(
                map_pool=map_pool,
                opponent_policy=opp,
                co_p0=args.co_p0,
                co_p1=args.co_p1,
                tier_name=args.tier,
                curriculum_tag="bo3_playoff",
            )
            if max_env_steps is not None:
                env_kw["max_env_steps"] = max_env_steps
            if args.max_p1_microsteps is not None:
                env_kw["max_p1_microsteps"] = int(args.max_p1_microsteps)
            if args.max_days is not None:
                env_kw["max_turns"] = int(args.max_days)
            env = ActionMasker(
                AWBWEnv(**env_kw),
                lambda e: e.action_masks(),
            )
            p0_model = load_maskable_ppo_compat(str(ch_snap), device="cpu", cache=True)
            while cw < args.first_to and dw < args.first_to and games < args.max_games:
                games += 1
                seed = int(rng.integers(0, 2**31 - 1))
                w = _play_one_game(
                    env, p0_model, rng_seed=seed, deterministic=args.deterministic
                )
                print(f"[bo3] game {games} winner={w}")
                if w == 0:
                    cw += 1
                elif w == 1:
                    dw += 1

        print(f"[bo3] challenger={cw} defender={dw} games={games}")
        if cw >= args.first_to:
            print("[bo3] Challenger wins the series.")
            if args.dry_run:
                print("[bo3] --dry-run: not replacing defender.")
            else:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                bk = df.parent / f"latest_pre_bo3_{ts}.zip"
                shutil.copy2(df, bk)
                shutil.copy2(ch, df)
                print(f"[bo3] backed up defender to {bk}; wrote challenger over {df}")
        else:
            print("[bo3] Defender wins the series (no file changes).")
    finally:
        clear_maskable_ppo_load_cache()
        if snap_paths:
            delete_eval_snapshots(snap_paths)


if __name__ == "__main__":
    main()
