"""
Random-vs-random self-play fuzzer: uniform picks from get_legal_actions(state).

Does not modify the engine. See desync_purge_engine_harden Phase 4 Thread FUZZER.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Repo root on sys.path when run as python tools/self_play_fuzzer.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.action import get_legal_actions  # noqa: E402
from engine.game import make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402

# Andy — keep CO complexity out of the fuzzer (campaign default).
_DEFAULT_CO_ID = 1

STEP_TIMEOUT_SEC = 5.0
GAME_TIMEOUT_SEC = 60.0
INVARIANT_EVERY_N_STEPS = 50


def _default_paths() -> tuple[Path, Path]:
    from server.play_human import MAPS_DIR, POOL_PATH

    return POOL_PATH, MAPS_DIR


def load_map_pool_ids(map_pool_path: Path) -> list[int]:
    with open(map_pool_path, encoding="utf-8") as f:
        pool: list[dict] = json.load(f)
    return [int(entry["map_id"]) for entry in pool]


def state_digest(state) -> dict[str, Any]:
    from engine.action import ActionStage

    return {
        "turn": state.turn,
        "active_player": state.active_player,
        "action_stage": state.action_stage.name
        if isinstance(state.action_stage, ActionStage)
        else str(state.action_stage),
        "done": state.done,
        "n_units": [len(state.units[0]), len(state.units[1])],
    }


def check_state_invariants(state) -> list[str]:
    """Lightweight checks if GameState has no validate(). Returns error strings."""
    errs: list[str] = []
    h, w = state.map_data.height, state.map_data.width

    for pl in (0, 1):
        seen: set[tuple[int, int]] = set()
        for u in state.units[pl]:
            if not u.is_alive:
                continue
            r, c = u.pos
            if not (0 <= r < h and 0 <= c < w):
                errs.append(f"P{pl} unit {u.unit_type.name} off-map at {u.pos}")
                continue
            if u.pos in seen:
                errs.append(f"P{pl} duplicate tile {u.pos}")
            seen.add(u.pos)
            if u.hp < 0:
                errs.append(f"P{pl} unit negative hp at {u.pos}")

    for pl in (0, 1):
        if state.funds[pl] < 0:
            errs.append(f"P{pl} negative funds {state.funds[pl]}")

    if state.active_player not in (0, 1):
        errs.append(f"active_player {state.active_player} not in {{0,1}}")

    # Turn / active consistency: odd turn with both players still playing is normal;
    # we only assert neither side has a corrupt CO reference.
    for pl in (0, 1):
        co = state.co_states[pl]
        if co is None:
            errs.append(f"P{pl} co_states is None")

    for pl in (0, 1):
        for u in state.units[pl]:
            if not u.is_alive:
                continue
            if u.fuel < 0 or u.ammo < 0:
                errs.append(f"P{pl} {u.unit_type.name} negative fuel/ammo")

    return errs


@dataclass
class Defect:
    type: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameResult:
    game_index: int
    seed: int
    map_id: int
    days_played: int
    actions_taken: int
    ended_normally: bool
    defects: list[Defect]
    duration_sec: float


def play_one_game(
    map_id: int,
    game_index: int,
    base_seed: int,
    max_days: int,
    pool_path: Path,
    maps_dir: Path,
    *,
    step_timeout_sec: float = STEP_TIMEOUT_SEC,
    game_timeout_sec: float = GAME_TIMEOUT_SEC,
) -> GameResult:
    rng = random.Random((int(base_seed) << 32) | (int(game_index) & 0xFFFFFFFF))

    defects: list[Defect] = []
    t0 = time.perf_counter()
    map_data = load_map(map_id, pool_path, maps_dir)
    state = make_initial_state(
        map_data,
        _DEFAULT_CO_ID,
        _DEFAULT_CO_ID,
        tier_name="T2",
    )
    actions_taken = 0
    ended_normally = False
    day_capped = False

    try:
        while not state.done:
            if state.turn > max_days:
                day_capped = True
                break
            if time.perf_counter() - t0 > game_timeout_sec:
                defects.append(
                    Defect("game_timeout", {"limit_sec": game_timeout_sec})
                )
                break

            legal = get_legal_actions(state)
            if not legal:
                defects.append(
                    Defect(
                        "empty_legal_set",
                        {"digest": state_digest(state)},
                    )
                )
                break

            action = rng.choice(legal)
            digest_before = state_digest(state)

            t_step = time.perf_counter()
            try:
                state.step(action)
            except BaseException as e:
                elapsed = time.perf_counter() - t_step
                if elapsed > step_timeout_sec:
                    defects.append(
                        Defect(
                            "step_timeout",
                            {
                                "elapsed_sec": elapsed,
                                "action": repr(action),
                            },
                        )
                    )
                defects.append(
                    Defect(
                        "mask_step_disagree",
                        {
                            "digest": digest_before,
                            "action": repr(action),
                            "exception": repr(e),
                            "exception_class": type(e).__name__,
                            "action_type": action.action_type.name,
                        },
                    )
                )
                actions_taken += 1
                continue

            elapsed = time.perf_counter() - t_step
            if elapsed > step_timeout_sec:
                defects.append(
                    Defect(
                        "step_timeout",
                        {
                            "elapsed_sec": elapsed,
                            "action": repr(action),
                        },
                    )
                )

            actions_taken += 1
            if actions_taken > 0 and actions_taken % INVARIANT_EVERY_N_STEPS == 0:
                inv_errs = check_state_invariants(state)
                if inv_errs:
                    defects.append(
                        Defect(
                            "invariant_violation",
                            {"errors": inv_errs, "digest": state_digest(state)},
                        )
                    )
            if state.done:
                ended_normally = True
                break
    except BaseException as e:
        defects.append(
            Defect(
                "uncaught_exception",
                {"error": f"{type(e).__name__}: {e}"},
            )
        )

    duration = time.perf_counter() - t0
    td = int(state.turn)
    if day_capped:
        td = min(td, max_days)
    return GameResult(
        game_index=game_index,
        seed=base_seed,
        map_id=map_id,
        days_played=td,
        actions_taken=actions_taken,
        ended_normally=ended_normally,
        defects=defects,
        duration_sec=duration,
    )


def _jsonl_row(gr: GameResult) -> dict[str, Any]:
    return {
        "game_index": gr.game_index,
        "seed": gr.seed,
        "map_id": gr.map_id,
        "days_played": gr.days_played,
        "actions_taken": gr.actions_taken,
        "ended_normally": gr.ended_normally,
        "defects": [{"type": d.type, **d.detail} for d in gr.defects],
        "duration_sec": gr.duration_sec,
    }


def run_fuzzer(
    games: int,
    seed: int,
    map_pool_path: Path,
    max_days: int,
    *,
    map_sample: Optional[int] = None,
    maps_dir: Optional[Path] = None,
    pool_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    quiet: bool = False,
    step_timeout_sec: float = STEP_TIMEOUT_SEC,
    game_timeout_sec: float = GAME_TIMEOUT_SEC,
) -> dict[str, Any]:
    """
    Run N random-vs-random games. Returns summary dict and writes JSONL if out_path set.

    Programmatic API for tests: inspect ``defects_by_type``, ``results``.
    """
    pool_path = pool_path or map_pool_path
    if maps_dir is None:
        _, default_maps = _default_paths()
        maps_dir = default_maps

    all_ids = load_map_pool_ids(map_pool_path)
    if not all_ids:
        raise ValueError(f"No map ids in {map_pool_path}")

    if map_sample is not None:
        k = min(int(map_sample), len(all_ids))
        rotation_pool = all_ids[:k]
    else:
        rotation_pool = list(all_ids)

    rng_global = random.Random(seed)
    results: list[GameResult] = []
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8") if out_path else None

    try:
        for g in range(games):
            if map_sample is not None:
                mid = rotation_pool[g % len(rotation_pool)]
            else:
                mid = rng_global.choice(rotation_pool)

            gr = play_one_game(
                mid,
                g,
                seed,
                max_days,
                pool_path,
                maps_dir,
                step_timeout_sec=step_timeout_sec,
                game_timeout_sec=game_timeout_sec,
            )
            results.append(gr)
            row = _jsonl_row(gr)
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()

    defect_counts: Counter[str] = Counter()
    disagree_shapes: Counter[str] = Counter()
    for gr in results:
        for d in gr.defects:
            defect_counts[d.type] += 1
            if d.type == "mask_step_disagree":
                exc_cls = d.detail.get("exception_class", "?")
                atn = d.detail.get("action_type") or "?"
                disagree_shapes[f"{atn}:{exc_cls}"] += 1

    summary = {
        "games": games,
        "seed": seed,
        "max_days": max_days,
        "map_sample": map_sample,
        "defects_by_type": dict(defect_counts),
        "top_mask_step_disagree_shapes": disagree_shapes.most_common(5),
        "results": results,
    }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    # stdout so shells (e.g. PowerShell) do not treat summary as stderr errors.
    print("--- summary ---")
    print(f"games: {summary['games']}")
    print(f"defects_by_type: {summary['defects_by_type']}")
    top = summary.get("top_mask_step_disagree_shapes") or []
    print("top 5 mask_step_disagree shapes:")
    for shape, cnt in top:
        print(f"  {cnt}  {shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AWBW random self-play fuzzer")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--map-pool",
        type=Path,
        default=_ROOT / "data" / "gl_map_pool.json",
    )
    parser.add_argument("--max-days", type=int, required=True)
    parser.add_argument("--map-sample", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from datetime import datetime

    if args.out is None:
        logs = _ROOT / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        args.out = logs / f"fuzzer_run_{datetime.now().strftime('%Y%m%d')}.jsonl"

    summary = run_fuzzer(
        args.games,
        args.seed,
        args.map_pool,
        args.max_days,
        map_sample=args.map_sample,
        out_path=args.out,
        quiet=args.quiet,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
