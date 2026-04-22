#!/usr/bin/env python3
"""
Fixed-length head-to-head on one map: ``--candidate`` plays P0 (challenger) for
``--games-first-seat`` games vs ``--baseline`` as P1, then zips swap roles for
``--games-second-seat`` games. Reports how often **candidate** wins.

Uses the same env wiring as ``scripts/bo3_checkpoint_playoff.py`` (no file writes).

At startup, copies both zips to ``<repo>/.tmp/eval_snap_*`` so mid-run overwrites
of shared paths (e.g. ``Z:\\latest.zip``) do not change the policies under test.

Example (Misery Andy mirror, ~7 games)::

  python scripts/symmetric_checkpoint_eval.py \\
    --candidate checkpoints/amarriner_bc.zip --baseline checkpoints/latest.zip \\
    --map-id 123858 --tier T3 --co-p0 1 --co-p1 1 \\
    --games-first-seat 4 --games-second-seat 3 --seed 0 \\
    --max-env-steps 0 --max-turns 150

Use ``--max-env-steps 0`` for unlimited P0 steps per game (otherwise the default
cap can truncate before a natural winner). Use ``--max-turns`` to raise the
engine calendar tiebreak above ``engine.game.MAX_TURNS`` (default 100).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _worker_game(payload: dict) -> tuple[int, int, bool]:
    """Return (game_index, winner, truncated_flag)."""
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

    class _Opp:
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
        curriculum_tag="symmetric_eval",
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
    obs, _ = env.reset(seed=seed)
    done = False
    last_trunc = False
    while not done:
        mask = env.action_masks()
        act, _ = p0.predict(obs, action_masks=mask, deterministic=bool(deterministic))
        obs, _r, term, trunc, info = env.step(int(act))
        last_trunc = bool(trunc or info.get("truncated"))
        done = bool(term or trunc)
    w_raw = env.unwrapped.state.winner
    w = -1 if w_raw is None else int(w_raw)
    return game_i, w, last_trunc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True, help="Fork / BC zip (evaluated as P0 then P1)")
    ap.add_argument("--baseline", type=Path, required=True, help="Baseline zip e.g. latest.zip")
    ap.add_argument("--map-id", type=int, required=True)
    ap.add_argument("--tier", type=str, required=True)
    ap.add_argument("--co-p0", type=int, required=True)
    ap.add_argument("--co-p1", type=int, required=True)
    ap.add_argument("--games-first-seat", type=int, default=4, help="Games with candidate as P0")
    ap.add_argument("--games-second-seat", type=int, default=3, help="Games with candidate as P1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None, help="Write summary JSON for findings")
    ap.add_argument(
        "--max-env-steps",
        type=int,
        default=100,
        help="Max P0 env.step calls per game (0 = unlimited). Prevents runaway episodes.",
    )
    ap.add_argument(
        "--max-p1-microsteps",
        type=int,
        default=None,
        help="Override cap on P1 engine steps per P0 step (default: derived from max-env-steps).",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=None,
        metavar="N",
        help="Engine calendar tiebreak: end-of-game property count after N days (default: MAX_TURNS from engine.game, usually 100).",
    )
    ap.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run each seating block in parallel processes (default: false). Each worker loads a full policy "
            "(multi-GB); use only if you have RAM for --parallel-workers concurrent loads (default cap: 2)."
        ),
    )
    ap.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max worker processes per seating block. Default caps at min(CPU, 2) because each worker "
            "loads a full policy (multi-GB); raise only if you have RAM headroom."
        ),
    )
    ap.add_argument(
        "--phi-profile",
        choices=("balanced", "capture"),
        default="balanced",
        help="Φ preset (α,β,κ) when using phi shaping; default balanced. Sets AWBW_PHI_PROFILE for this run.",
    )
    args = ap.parse_args()
    # Must be set before any import that loads engine.game (e.g. rl.self_play); workers inherit the env.
    os.environ["AWBW_PHI_PROFILE"] = str(args.phi_profile)

    max_env_steps = (
        None if args.max_env_steps is None or args.max_env_steps <= 0 else int(args.max_env_steps)
    )
    if args.max_turns is not None:
        if int(args.max_turns) < 1:
            raise SystemExit("--max-turns must be >= 1")

    cand_src, base_src = args.candidate.resolve(), args.baseline.resolve()
    if not cand_src.is_file():
        raise SystemExit(f"Missing candidate: {cand_src}")
    if not base_src.is_file():
        raise SystemExit(f"Missing baseline: {base_src}")

    from rl.ckpt_compat import (
        clear_maskable_ppo_load_cache,
        delete_eval_snapshots,
        snapshot_eval_checkpoints,
    )

    snap_paths: tuple[Path, ...] | None = None
    try:
        run_id, (cand_snap, base_snap) = snapshot_eval_checkpoints(
            [("candidate", cand_src), ("baseline", base_src)]
        )
        snap_paths = (cand_snap, base_snap)
        print(f"[sym] frozen snapshots run_id={run_id}")
        print(f"[sym]   candidate -> {cand_snap}")
        print(f"[sym]   baseline -> {base_snap}")

        from rl.self_play import POOL_PATH

        import numpy as np

        with Path(POOL_PATH).open(encoding="utf-8") as f:
            map_pool = json.load(f)
        map_pool = [m for m in map_pool if m.get("map_id") == args.map_id]
        if not map_pool:
            raise SystemExit("no maps for --map-id")
        pool_json = json.dumps(map_pool)
        rng = np.random.default_rng(args.seed)

        results: list[dict] = []
        cand_wins = 0
        base_wins = 0
        game_no = 0

        def _one_game_outcome(
            *,
            challenger: Path,
            defender: Path,
            game_no: int,
            seed: int,
            label: str,
        ) -> None:
            nonlocal cand_wins, base_wins
            _gi, w, was_trunc = _worker_game(
                {
                    "game_i": game_no,
                    "seed": seed,
                    "challenger": str(challenger),
                    "defender": str(defender),
                    "map_pool_json": pool_json,
                    "tier": args.tier,
                    "co_p0": args.co_p0,
                    "co_p1": args.co_p1,
                    "deterministic": args.deterministic,
                    "max_env_steps": max_env_steps,
                    "max_p1_microsteps": args.max_p1_microsteps,
                    "max_turns": args.max_turns,
                }
            )
            if challenger == cand_snap:
                cw = w == 0
                bw = w == 1
            else:
                cw = w == 1
                bw = w == 0
            if cw:
                cand_wins += 1
            elif bw:
                base_wins += 1
            print(
                f"[sym] game={game_no} block={label} seed={seed} winner_p0={w} "
                f"candidate_win={cw} truncated={was_trunc}"
            )
            results.append(
                {
                    "game": game_no,
                    "block": label,
                    "seed": seed,
                    "winner": w,
                    "candidate_win": cw,
                    "truncated": was_trunc,
                }
            )

        def run_block(
            *,
            challenger: Path,
            defender: Path,
            n_games: int,
            label: str,
        ) -> None:
            nonlocal game_no, cand_wins, base_wins
            if n_games <= 0:
                return
            tasks: list[tuple[int, int]] = []
            for _ in range(n_games):
                game_no += 1
                seed = int(rng.integers(0, 2**31 - 1))
                tasks.append((game_no, seed))

            cpu = os.cpu_count() or 4
            if args.parallel_workers is not None:
                cap = int(args.parallel_workers)
            else:
                # Uncapped parallel (e.g. one worker per game) can OOM: each process loads MaskablePPO + buffers.
                cap = min(cpu, 2)
            workers = max(1, min(n_games, cap))
            use_parallel = bool(args.parallel) and n_games > 1 and workers > 1
            if use_parallel:
                payloads = [
                    {
                        "game_i": gn,
                        "seed": sd,
                        "challenger": str(challenger),
                        "defender": str(defender),
                        "map_pool_json": pool_json,
                        "tier": args.tier,
                        "co_p0": args.co_p0,
                        "co_p1": args.co_p1,
                        "deterministic": args.deterministic,
                        "max_env_steps": max_env_steps,
                        "max_p1_microsteps": args.max_p1_microsteps,
                        "max_turns": args.max_turns,
                    }
                    for gn, sd in tasks
                ]
                print(f"[sym] parallel block={label} workers={workers} games={n_games}")
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_worker_game, p): p for p in payloads}
                    by_i: dict[int, tuple[int, bool]] = {}
                    for fut in as_completed(futs):
                        p = futs[fut]
                        gi, w, was_trunc = fut.result()
                        by_i[int(gi)] = (w, was_trunc)
                for gn, sd in tasks:
                    w, was_trunc = by_i[gn]
                    if challenger == cand_snap:
                        cw = w == 0
                        bw = w == 1
                    else:
                        cw = w == 1
                        bw = w == 0
                    if cw:
                        cand_wins += 1
                    elif bw:
                        base_wins += 1
                    print(
                        f"[sym] game={gn} block={label} seed={sd} winner_p0={w} "
                        f"candidate_win={cw} truncated={was_trunc}"
                    )
                    results.append(
                        {
                            "game": gn,
                            "block": label,
                            "seed": sd,
                            "winner": w,
                            "candidate_win": cw,
                            "truncated": was_trunc,
                        }
                    )
            else:
                for gn, sd in tasks:
                    _one_game_outcome(
                        challenger=challenger,
                        defender=defender,
                        game_no=gn,
                        seed=sd,
                        label=label,
                    )

        print(
            f"[sym] candidate_as_P0 x{args.games_first_seat} "
            f"then candidate_as_P1 x{args.games_second_seat} "
            f"(max_env_steps={max_env_steps}, max_turns={args.max_turns}, "
            f"parallel={args.parallel})"
        )
        run_block(
            challenger=cand_snap,
            defender=base_snap,
            n_games=args.games_first_seat,
            label="candidate_P0",
        )
        run_block(
            challenger=base_snap,
            defender=cand_snap,
            n_games=args.games_second_seat,
            label="candidate_P1",
        )

        results.sort(key=lambda r: int(r["game"]))

        total = cand_wins + base_wins
        print(
            f"[sym] summary candidate_wins={cand_wins} baseline_wins={base_wins} "
            f"total_decided={total} (draws if winner==-1 not counted above)"
        )
        if total:
            print(f"[sym] candidate_win_rate={cand_wins / total:.3f}")

        # Per-seat candidate win counts
        w_p0 = sum(1 for r in results if r["block"] == "candidate_P0" and r["candidate_win"])
        n_p0 = sum(1 for r in results if r["block"] == "candidate_P0")
        w_p1 = sum(1 for r in results if r["block"] == "candidate_P1" and r["candidate_win"])
        n_p1 = sum(1 for r in results if r["block"] == "candidate_P1")
        print(f"[sym] as_P0: candidate {w_p0}/{n_p0}  as_P1: candidate {w_p1}/{n_p1}")

        promote_ok = cand_wins > base_wins and (n_p1 == 0 or w_p1 > 0) and (n_p0 == 0 or w_p0 > 0)
        print(
            f"[sym] promotion_heuristic_ok={promote_ok} "
            "(candidate ahead overall; no 0-for-N collapse in either seat)"
        )

        if args.json_out:
            summary = {
                "candidate": str(cand_src),
                "baseline": str(base_src),
                "candidate_snapshot": str(cand_snap),
                "baseline_snapshot": str(base_snap),
                "eval_snapshot_run_id": run_id,
                "parallel": bool(args.parallel),
                "parallel_workers": args.parallel_workers,
                "map_id": args.map_id,
                "tier": args.tier,
                "max_env_steps": max_env_steps,
                "max_turns": args.max_turns,
                "max_p1_microsteps": args.max_p1_microsteps,
                "co_p0": args.co_p0,
                "co_p1": args.co_p1,
                "phi_profile": str(args.phi_profile),
                "candidate_wins": cand_wins,
                "baseline_wins": base_wins,
                "per_seat": {"candidate_as_p0": [w_p0, n_p0], "candidate_as_p1": [w_p1, n_p1]},
                "games": results,
                "promotion_heuristic_ok": promote_ok,
            }
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[sym] wrote {args.json_out}")

        return 0
    finally:
        clear_maskable_ppo_load_cache()
        if snap_paths:
            delete_eval_snapshots(snap_paths)


if __name__ == "__main__":
    raise SystemExit(main())
