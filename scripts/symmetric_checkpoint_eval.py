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
import copy
import json
import logging
import os
import statistics
import sys
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_LOG = logging.getLogger(__name__)


def build_symmetric_checkpoint_eval_parser() -> argparse.ArgumentParser:
    """Argparse for symmetric eval (used by main and tests)."""
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
    ap.add_argument(
        "--mcts-mode",
        type=str,
        default="off",
        choices=("off", "eval_only"),
        help=(
            "Phase 11c: turn-boundary MCTS for P0 only (challenger zip). "
            "P1 keeps the existing opponent policy. Default off."
        ),
    )
    ap.add_argument("--mcts-sims", type=int, default=16, help="MCTS simulations per root.")
    ap.add_argument("--mcts-c-puct", type=float, default=1.5, help="PUCT c_puct.")
    ap.add_argument("--mcts-dirichlet-alpha", type=float, default=0.3, help="Root Dirichlet alpha.")
    ap.add_argument("--mcts-dirichlet-epsilon", type=float, default=0.25, help="Root Dirichlet mix.")
    ap.add_argument("--mcts-temperature", type=float, default=1.0, help="Root plan selection temperature.")
    ap.add_argument("--mcts-min-depth", type=int, default=4, help="Min depth before PUCT.")
    ap.add_argument("--mcts-root-plans", type=int, default=8, help="Root expansion plan samples.")
    ap.add_argument(
        "--mcts-max-plan-actions",
        type=int,
        default=256,
        help="Max actions per simulated turn in MCTS expansion.",
    )
    return ap


def _mcts_fields_from_args(ns: argparse.Namespace) -> dict:
    """Subset passed to workers (JSON-serializable)."""
    return {
        "mcts_mode": str(ns.mcts_mode),
        "mcts_sims": int(ns.mcts_sims),
        "mcts_c_puct": float(ns.mcts_c_puct),
        "mcts_dirichlet_alpha": float(ns.mcts_dirichlet_alpha),
        "mcts_dirichlet_epsilon": float(ns.mcts_dirichlet_epsilon),
        "mcts_temperature": float(ns.mcts_temperature),
        "mcts_min_depth": int(ns.mcts_min_depth),
        "mcts_root_plans": int(ns.mcts_root_plans),
        "mcts_max_plan_actions": int(ns.mcts_max_plan_actions),
    }


def _worker_game(payload: dict) -> tuple[int, int, bool, dict]:
    """Return (game_index, winner, truncated_flag, mcts_telemetry)."""
    import numpy as np
    from sb3_contrib.common.wrappers import ActionMasker

    from engine.action import ActionStage, ActionType

    from rl.ckpt_compat import load_maskable_ppo_compat
    from rl.env import AWBWEnv, _action_to_flat, _flat_to_action
    from rl.mcts import MCTSConfig, make_callables_from_sb3_policy, run_mcts

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

    mcts_mode = str(payload.get("mcts_mode") or "off").strip().lower()
    mcts_telemetry: dict = {
        "mcts_decision_wall_s": [],
        "mcts_pv_depths": [],
        "mcts_total_wall_s": 0.0,
        "mcts_failures": 0,
        "mcts_total_decisions": 0,
    }
    mcts_cfg = MCTSConfig(
        num_sims=int(payload.get("mcts_sims", 16)),
        c_puct=float(payload.get("mcts_c_puct", 1.5)),
        dirichlet_alpha=float(payload.get("mcts_dirichlet_alpha", 0.3)),
        dirichlet_epsilon=float(payload.get("mcts_dirichlet_epsilon", 0.25)),
        temperature=float(payload.get("mcts_temperature", 1.0)),
        min_depth=int(payload.get("mcts_min_depth", 4)),
        root_plans=int(payload.get("mcts_root_plans", 8)),
        max_plan_actions=int(payload.get("mcts_max_plan_actions", 256)),
    )
    use_mcts = mcts_mode == "eval_only" and mcts_cfg.num_sims > 0
    zero_warned = False

    def _pred_int(act: object) -> int:
        if isinstance(act, int | np.integer):
            return int(act)
        return int(np.asarray(act, dtype=np.int64).reshape(-1)[0])

    class _Opp:
        def __init__(self) -> None:
            self._m = None

        def __call__(self, obs, mask):
            if self._m is None:
                self._m = load_maskable_ppo_compat(str(df), device="cpu", cache=True)
            a, _ = self._m.predict(obs, action_masks=mask, deterministic=deterministic)
            return _pred_int(a)

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

    uenv = env.unwrapped
    cached_plan: deque = deque()
    mid_turn_plan_exhausted = False
    mcts_decision_ix = 0

    def _p0_flat_direct() -> int:
        m = env.action_masks()
        act, _ = p0.predict(obs, action_masks=m, deterministic=bool(deterministic))
        return _pred_int(act)

    while not done:
        st = uenv.state
        act_idx: int | None = None
        chosen_by_mcts_queue = False

        if mcts_mode == "eval_only" and mcts_cfg.num_sims == 0 and not zero_warned:
            _LOG.warning(
                "[sym] --mcts-mode eval_only but mcts_sims=0; using direct policy for P0"
            )
            zero_warned = True

        if not use_mcts:
            act_idx = _p0_flat_direct()
        elif len(cached_plan) > 0:
            act_obj = cached_plan.popleft()
            act_idx = _action_to_flat(act_obj, st)
            chosen_by_mcts_queue = True
        elif mid_turn_plan_exhausted:
            act_idx = _p0_flat_direct()
        elif st.action_stage == ActionStage.SELECT:
            try:
                root = copy.deepcopy(st)
                pol_c, val_c, prior_c = make_callables_from_sb3_policy(p0, uenv)
                mcts_cfg_run = MCTSConfig(
                    num_sims=mcts_cfg.num_sims,
                    c_puct=mcts_cfg.c_puct,
                    dirichlet_alpha=mcts_cfg.dirichlet_alpha,
                    dirichlet_epsilon=mcts_cfg.dirichlet_epsilon,
                    temperature=mcts_cfg.temperature,
                    min_depth=mcts_cfg.min_depth,
                    root_plans=mcts_cfg.root_plans,
                    max_plan_actions=mcts_cfg.max_plan_actions,
                    rng_seed=int(seed) ^ (mcts_decision_ix * 0x9E3779B9),
                )
                mcts_decision_ix += 1
                plan, stats = run_mcts(
                    root,
                    policy_callable=pol_c,
                    value_callable=val_c,
                    prior_callable=prior_c,
                    config=mcts_cfg_run,
                )
                mcts_telemetry["mcts_decision_wall_s"].append(float(stats["wall_time_s"]))
                mcts_telemetry["mcts_pv_depths"].append(int(stats["principal_variation_depth"]))
                mcts_telemetry["mcts_total_wall_s"] += float(stats["wall_time_s"])
                mcts_telemetry["mcts_total_decisions"] += 1
                if plan:
                    cached_plan.extend(plan)
            except Exception:
                mcts_telemetry["mcts_failures"] += 1
                _LOG.error("[sym] MCTS failed; falling back to direct policy for this step\n%s", traceback.format_exc())
                cached_plan.clear()
                mid_turn_plan_exhausted = False
                act_idx = _p0_flat_direct()

            if act_idx is None and len(cached_plan) > 0:
                act_obj = cached_plan.popleft()
                act_idx = _action_to_flat(act_obj, st)
                chosen_by_mcts_queue = True
            elif act_idx is None:
                act_idx = _p0_flat_direct()
        else:
            act_idx = _p0_flat_direct()

        st_pre = uenv.state
        obs, _r, term, trunc, info = env.step(int(act_idx))
        last_trunc = bool(trunc or info.get("truncated"))
        done = bool(term or trunc)

        dec = _flat_to_action(int(act_idx), st_pre, legal=uenv._get_legal())
        st_after = uenv.state
        if len(cached_plan) > 0:
            mid_turn_plan_exhausted = False
        elif dec is not None and dec.action_type == ActionType.END_TURN:
            mid_turn_plan_exhausted = False
        elif (
            chosen_by_mcts_queue
            and use_mcts
            and not (term or trunc)
            and st_after.active_player == 0
        ):
            # Documented: MCTS plan ended before END_TURN; remaining P0 actions use direct policy.
            mid_turn_plan_exhausted = True

    w_raw = env.unwrapped.state.winner
    w = -1 if w_raw is None else int(w_raw)
    return game_i, w, last_trunc, mcts_telemetry


def main() -> int:
    ap = build_symmetric_checkpoint_eval_parser()
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
        mcts_payload = _mcts_fields_from_args(args)

        results: list[dict] = []
        cand_wins = 0
        base_wins = 0
        game_no = 0
        agg_mcts_walls: list[float] = []
        agg_mcts_pv: list[int] = []
        agg_mcts_failures = 0
        agg_mcts_decisions = 0
        agg_mcts_total_wall = 0.0

        def _accumulate_mcts_tel(tel: dict) -> None:
            nonlocal agg_mcts_failures, agg_mcts_decisions, agg_mcts_total_wall
            agg_mcts_walls.extend(float(x) for x in tel.get("mcts_decision_wall_s", []))
            agg_mcts_pv.extend(int(x) for x in tel.get("mcts_pv_depths", []))
            agg_mcts_failures += int(tel.get("mcts_failures", 0))
            agg_mcts_decisions += int(tel.get("mcts_total_decisions", 0))
            agg_mcts_total_wall += float(tel.get("mcts_total_wall_s", 0.0))

        def _mk_worker_payload(
            gn: int,
            sd: int,
            challenger: Path,
            defender: Path,
        ) -> dict:
            return {
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
                **mcts_payload,
            }

        def _one_game_outcome(
            *,
            challenger: Path,
            defender: Path,
            game_no: int,
            seed: int,
            label: str,
        ) -> None:
            nonlocal cand_wins, base_wins
            _gi, w, was_trunc, tel = _worker_game(_mk_worker_payload(game_no, seed, challenger, defender))
            _accumulate_mcts_tel(tel)
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
                    _mk_worker_payload(gn, sd, challenger, defender) for gn, sd in tasks
                ]
                print(f"[sym] parallel block={label} workers={workers} games={n_games}")
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_worker_game, p): p for p in payloads}
                    by_i: dict[int, tuple[int, bool, dict]] = {}
                    for fut in as_completed(futs):
                        p = futs[fut]
                        gi, w, was_trunc, tel = fut.result()
                        by_i[int(gi)] = (w, was_trunc, tel)
                for gn, sd in tasks:
                    w, was_trunc, tel = by_i[gn]
                    _accumulate_mcts_tel(tel)
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
            f"parallel={args.parallel}, mcts_mode={args.mcts_mode})"
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
            if args.mcts_mode == "eval_only":
                summary["mcts_per_decision_wall_s_p50"] = (
                    float(statistics.median(agg_mcts_walls)) if agg_mcts_walls else None
                )
                summary["mcts_total_decisions"] = int(agg_mcts_decisions)
                summary["mcts_total_wall_s"] = float(agg_mcts_total_wall)
                summary["mcts_avg_principal_variation_depth"] = (
                    float(sum(agg_mcts_pv) / len(agg_mcts_pv)) if agg_mcts_pv else None
                )
                summary["mcts_failure_count"] = int(agg_mcts_failures)
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
