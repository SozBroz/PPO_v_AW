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
    --max-env-steps 0 --max-days 150

Default ``--max-env-steps`` is **10000** P0 steps per game (aligned with fleet training caps); use
``0`` for unlimited. Use ``--max-days`` (alias ``--max-turns``, deprecated) to raise the engine
end-inclusive calendar tiebreak above
``engine.game.MAX_TURNS`` (default 100).

By default all games run **concurrently** (both seating blocks in one pool).
At most **7** worker processes run at once; use ``--no-parallel`` for sequential.

With ``--turn-heartbeat`` (default on), each worker prints a line whenever the
engine calendar day advances: funds, income-property counts, total owned
properties, and unit counts — so long parallel games show progress.
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

from rl.ai_vs_ai import _MaxCalendarDaysAction

_LOG = logging.getLogger(__name__)

# Each process loads a full MaskablePPO (~multi-GB). Never exceed this concurrency.
_SYMMETRIC_EVAL_MAX_CONCURRENT_WORKERS = 7


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
        "--turn-heartbeat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Per worker: print a line when the engine calendar day advances (funds, income props, "
            "owned props, units). Use --no-turn-heartbeat for quiet output."
        ),
    )
    ap.add_argument(
        "--max-env-steps",
        type=int,
        default=10000,
        help="Max P0 env.step calls per game (0 = unlimited). Default 10000; prevents runaway episodes.",
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
        help=(
            "End-inclusive engine calendar day cap / property tiebreak "
            "(default: engine MAX_TURNS, usually 100)."
        ),
    )
    ap.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run all eval games in parallel worker processes (default: true). Both seating blocks share "
            "one pool with at most "
            f"{_SYMMETRIC_EVAL_MAX_CONCURRENT_WORKERS} concurrent workers. Each worker loads a full policy "
            "(multi-GB). Use --no-parallel to run sequentially."
        ),
    )
    ap.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Target max worker processes (default: CPU count). Hard-capped at "
            f"{_SYMMETRIC_EVAL_MAX_CONCURRENT_WORKERS} so more than that never run at once."
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
    ap.add_argument(
        "--mcts-rollout-stage",
        type=str,
        default=None,
        choices=("mcts_0", "mcts_1", "mcts_2", "mcts_3", "mcts_4"),
        help=(
            "MASTERPLAN §14 preset: mcts_0=plumbing, mcts_1=eval, mcts_2=selective P0, "
            "mcts_3=distillation-flavored defaults, mcts_4=anytime (sim cap+wall time). "
            "Omitted MCTS args use the preset; pass individual --mcts-* to override. "
            "See rl/mcts_rollout_stages.py."
        ),
    )
    ap.add_argument(
        "--mcts-sims",
        type=int,
        default=argparse.SUPPRESS,
        help="MCTS simulations per root (default: 16, or the rollout stage).",
    )
    ap.add_argument(
        "--mcts-c-puct", type=float, default=argparse.SUPPRESS, help="PUCT c_puct (default: 1.5 or stage)."
    )
    ap.add_argument(
        "--mcts-dirichlet-alpha", type=float, default=argparse.SUPPRESS, help="Root Dirichlet alpha (default: 0.3 or stage)."
    )
    ap.add_argument(
        "--mcts-dirichlet-epsilon",
        type=float,
        default=argparse.SUPPRESS,
        help="Root Dirichlet mix (default: 0.25 without stage, or stage).",
    )
    ap.add_argument(
        "--mcts-temperature", type=float, default=argparse.SUPPRESS, help="Root plan selection (default: 1.0 or stage)."
    )
    ap.add_argument("--mcts-min-depth", type=int, default=argparse.SUPPRESS, help="Min depth before PUCT (default: 4 or stage).")
    ap.add_argument("--mcts-root-plans", type=int, default=argparse.SUPPRESS, help="Root expansion plan samples (default: 8 or stage).")
    ap.add_argument(
        "--mcts-max-plan-actions",
        type=int,
        default=argparse.SUPPRESS,
        help="Max actions per simulated turn in MCTS expansion (default: 256).",
    )
    ap.add_argument(
        "--mcts-luck-resamples", type=int, default=argparse.SUPPRESS, help="Root luck resamples (default: 0 or stage)."
    )
    ap.add_argument(
        "--mcts-luck-resample-critical-only",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="If set, only resample on critical-trace children.",
    )
    ap.add_argument(
        "--mcts-risk-mode",
        type=str,
        default=argparse.SUPPRESS,
        choices=("visit", "mean", "mean_minus_p10", "constrained"),
        help="Root risk selection mode (default: visit or stage).",
    )
    ap.add_argument("--mcts-risk-lambda", type=float, default=argparse.SUPPRESS, help="Tail penalty (default: 0.35).")
    ap.add_argument("--mcts-catastrophe-value", type=float, default=argparse.SUPPRESS, help="Catastrophe value threshold (default: -0.35).")
    ap.add_argument(
        "--mcts-max-catastrophe-prob", type=float, default=argparse.SUPPRESS, help="Constrained mode max catastrophe prob (default: 1.0)."
    )
    ap.add_argument("--mcts-root-decision-log", type=str, default=argparse.SUPPRESS, help="JSONL for per-root stats.")
    ap.add_argument(
        "--mcts-max-wall-time-s",
        type=float,
        default=argparse.SUPPRESS,
        help="Max wall seconds for the main MCTS sim loop (MCTS-4 / overrides). None = off.",
    )
    ap.add_argument(
        "--mcts-p0-mcts-invocation-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of P0 SELECTs that run MCTS; rest use direct policy (MCTS-2 / overrides). 1.0=always.",
    )
    return ap


def _worker_game(payload: dict) -> tuple[int, int, bool, dict]:
    """Return (game_index, winner, truncated_flag, mcts_telemetry)."""
    import numpy as np
    from sb3_contrib.common.wrappers import ActionMasker

    from engine.action import ActionStage, ActionType

    from rl.ckpt_compat import load_maskable_ppo_compat
    from rl.env import AWBWEnv, _action_to_flat, _flat_to_action
    from dataclasses import replace

    from rl.mcts import (
        decision_log_context_from_env,
        make_callables_from_sb3_policy,
        run_mcts,
    )
    from rl.mcts_rollout_stages import mcts_config_from_eval_payload

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

    turn_heartbeat = bool(payload.get("turn_heartbeat", True))
    mcts_mode = str(payload.get("mcts_mode") or "off").strip().lower()
    mcts_telemetry: dict = {
        "mcts_decision_wall_s": [],
        "mcts_pv_depths": [],
        "mcts_total_wall_s": 0.0,
        "mcts_failures": 0,
        "mcts_total_decisions": 0,
        "mcts_root_entropy": [],
        "mcts_chosen_risk": [],
        "mcts_decision_log_context": [],
        "mcts_rollout_resolved": {},
    }
    mcts_cfg = mcts_config_from_eval_payload({**payload, "mcts_mode": mcts_mode})
    mcts_telemetry["mcts_rollout_resolved"] = {
        "mcts_rollout_stage": mcts_cfg.rollout_stage,
        "num_sims": mcts_cfg.num_sims,
        "max_wall_time_s": mcts_cfg.max_wall_time_s,
        "p0_mcts_invocation_fraction": mcts_cfg.p0_mcts_invocation_fraction,
    }
    use_mcts = mcts_mode == "eval_only" and mcts_cfg.num_sims > 0
    zero_warned = False
    p0_select_n = 0

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
    last_hb_day = uenv.state.turn
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
            p0_select_n += 1
            try:
                if (
                    use_mcts
                    and float(mcts_cfg.p0_mcts_invocation_fraction) < 1.0
                    and np.random.default_rng(
                        (int(seed) * 0x9E3779B9) ^ (p0_select_n * 0xC2B2AE3D)
                    ).random()
                    >= float(mcts_cfg.p0_mcts_invocation_fraction)
                ):
                    act_idx = _p0_flat_direct()
                else:
                    root = copy.deepcopy(st)
                    pol_c, val_c, prior_c = make_callables_from_sb3_policy(p0, uenv)
                    mcts_cfg_run = replace(
                        mcts_cfg,
                        rng_seed=int(seed) ^ (mcts_decision_ix * 0x9E3779B9),
                    )
                    mcts_decision_ix += 1
                    plan, stats = run_mcts(
                        root,
                        policy_callable=pol_c,
                        value_callable=val_c,
                        prior_callable=prior_c,
                        config=mcts_cfg_run,
                        decision_log_context=decision_log_context_from_env(uenv),
                    )
                    mcts_telemetry["mcts_decision_wall_s"].append(float(stats["wall_time_s"]))
                    mcts_telemetry["mcts_pv_depths"].append(int(stats["principal_variation_depth"]))
                    mcts_telemetry["mcts_total_wall_s"] += float(stats["wall_time_s"])
                    mcts_telemetry["mcts_total_decisions"] += 1
                    mcts_telemetry["mcts_root_entropy"].append(float(stats.get("root_visit_entropy", 0.0)))
                    mcts_telemetry["mcts_chosen_risk"].append(stats.get("chosen_risk", {}))
                    mcts_telemetry["mcts_decision_log_context"].append(
                        stats.get("decision_log_context", {})
                    )
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
        if turn_heartbeat and st_after.turn > last_hb_day:
            s = st_after
            p0i = s.count_income_properties(0)
            p1i = s.count_income_properties(1)
            p0a = s.count_properties(0)
            p1a = s.count_properties(1)
            u0 = len(s.units[0])
            u1 = len(s.units[1])
            print(
                f"[sym] hb g={game_i} day={s.turn} ap={s.active_player} "
                f"funds=({s.funds[0]},{s.funds[1]}) inc_props=({p0i},{p1i}) "
                f"props=({p0a},{p1a}) units=({u0},{u1})",
                flush=True,
            )
            last_hb_day = s.turn
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


def mcts_work_payload_from_argparse(ns: argparse.Namespace) -> dict:
    """Lazily import rl (engine) so this module stays safe for tests that set env first."""
    from rl.mcts_rollout_stages import mcts_work_payload_from_argparse as _impl

    return _impl(ns)


# Backward compat for tests: old name
_mcts_fields_from_args = mcts_work_payload_from_argparse


def main() -> int:
    ap = build_symmetric_checkpoint_eval_parser()
    args = ap.parse_args()
    # Must be set before any import that loads engine.game (e.g. rl.self_play); workers inherit the env.
    os.environ["AWBW_PHI_PROFILE"] = str(args.phi_profile)

    max_env_steps = (
        None if args.max_env_steps is None or args.max_env_steps <= 0 else int(args.max_env_steps)
    )
    if args.max_days is not None:
        if int(args.max_days) < 1:
            raise SystemExit("--max-days must be >= 1")

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
        mcts_payload = mcts_work_payload_from_argparse(args)

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
                "max_turns": args.max_days,
                "max_days": args.max_days,
                "turn_heartbeat": bool(args.turn_heartbeat),
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

        def _effective_parallel_workers(n_tasks: int) -> int:
            cpu = os.cpu_count() or 4
            if args.parallel_workers is not None:
                raw = max(1, int(args.parallel_workers))
            else:
                raw = max(1, int(cpu))
            return max(1, min(n_tasks, raw, _SYMMETRIC_EVAL_MAX_CONCURRENT_WORKERS))

        all_jobs: list[tuple[int, int, Path, Path, str]] = []
        for _ in range(args.games_first_seat):
            game_no += 1
            seed = int(rng.integers(0, 2**31 - 1))
            all_jobs.append((game_no, seed, cand_snap, base_snap, "candidate_P0"))
        for _ in range(args.games_second_seat):
            game_no += 1
            seed = int(rng.integers(0, 2**31 - 1))
            all_jobs.append((game_no, seed, base_snap, cand_snap, "candidate_P1"))

        n_tasks = len(all_jobs)
        workers = _effective_parallel_workers(n_tasks) if n_tasks else 1
        use_parallel = bool(args.parallel) and n_tasks > 1 and workers > 1

        print(
            f"[sym] candidate_as_P0 x{args.games_first_seat} "
            f"then candidate_as_P1 x{args.games_second_seat} "
            f"(max_env_steps={max_env_steps}, max_days={args.max_days}, "
            f"parallel={args.parallel}, parallel_workers={workers}, "
            f"parallel_workers_max={_SYMMETRIC_EVAL_MAX_CONCURRENT_WORKERS}, mcts_mode={args.mcts_mode})"
        )

        if use_parallel:
            payloads = [_mk_worker_payload(gn, sd, ch, df) for gn, sd, ch, df, _lb in all_jobs]
            print(f"[sym] parallel all_blocks workers={workers} games={n_tasks}")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_worker_game, p): p for p in payloads}
                by_i: dict[int, tuple[int, bool, dict]] = {}
                for fut in as_completed(futs):
                    p = futs[fut]
                    gi, w, was_trunc, tel = fut.result()
                    by_i[int(gi)] = (w, was_trunc, tel)
            for gn, sd, ch, df, label in all_jobs:
                w, was_trunc, tel = by_i[gn]
                _accumulate_mcts_tel(tel)
                if ch == cand_snap:
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
            for gn, sd, ch, df, label in all_jobs:
                _one_game_outcome(
                    challenger=ch,
                    defender=df,
                    game_no=gn,
                    seed=sd,
                    label=label,
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
                "parallel_workers": int(workers),
                "parallel_workers_max": _SYMMETRIC_EVAL_MAX_CONCURRENT_WORKERS,
                "map_id": args.map_id,
                "tier": args.tier,
                "max_env_steps": max_env_steps,
                "max_turns": args.max_days,
                "max_days": args.max_days,
                "max_p1_microsteps": args.max_p1_microsteps,
                "co_p0": args.co_p0,
                "co_p1": args.co_p1,
                "phi_profile": str(args.phi_profile),
                "turn_heartbeat": bool(args.turn_heartbeat),
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
