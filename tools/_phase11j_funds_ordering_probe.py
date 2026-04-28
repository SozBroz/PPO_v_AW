#!/usr/bin/env python3
"""Phase 11J-FUNDS-CORPUS — empirical income vs property-repair ordering from PHP zips.

At each ``_end_turn`` (start of opponent's turn), compares AWBW PHP ``players[*].funds``
(frame after the ``p:`` envelope that contained ``End``) to engine funds under:

  * **IBR** — income then repair (``_grant_income`` then ``_resupply_on_properties``).
  * **RBI** — repair then income (current engine order).

See ``docs/oracle_exception_audit/phase11j_funds_corpus_derivation.md``.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import ActionStage
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, idle_start_of_day_fuel_drain
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.replay_snapshot_compare import replay_snapshot_pairing

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"

# Mission filter: no Rachel (28), Hachi (17), Kindle (23) — shipped / special lanes.
EXCLUDED_CO_IDS = frozenset({17, 28, 23})

BIN_NAMES = (
    "PHP_MATCHES_IBR",
    "PHP_MATCHES_RBI",
    "PHP_MATCHES_NEITHER",
    "AMBIGUOUS",
)


def _frame_funds_by_engine_seat(frame: dict, awbw_to_engine: dict[int, int]) -> dict[int, int]:
    out = {0: 0, 1: 0}
    for _k, pl in (frame.get("players") or {}).items():
        try:
            pid = int(pl.get("id"))
        except (TypeError, ValueError):
            continue
        if pid in awbw_to_engine:
            try:
                out[int(awbw_to_engine[pid])] = int(pl.get("funds") or 0)
            except (TypeError, ValueError):
                pass
    return out


def _run_end_turn_prefix_to_property_resupply(gs: GameState) -> Optional[int]:
    """Mirror ``GameState._end_turn`` in ``engine/game.py`` through the fuel/crash
    block (exclusive of ``_resupply_on_properties``).

    Returns the opponent seat (``gs.active_player`` after the switch) when the
    normal path continues to property resupply + income; ``None`` when the game
    ends at the calendar cap before that (same early return as the engine).
    """
    player = gs.active_player
    opponent = 1 - player

    co = gs.co_states[player]
    co.cop_active = False
    co.scop_active = False

    if gs.co_weather_segments_remaining > 0:
        gs.co_weather_segments_remaining -= 1
        if gs.co_weather_segments_remaining == 0:
            gs.weather = gs.default_weather

    gs.active_player = opponent
    gs.action_stage = ActionStage.SELECT
    gs.selected_unit = None
    gs.selected_move_pos = None

    if opponent == 0:
        gs.turn += 1
        if gs.turn > gs.max_turns:
            gs.done = True
            p0_props = gs.count_properties(0)
            p1_props = gs.count_properties(1)
            if p0_props > p1_props:
                gs.winner = 0
                gs.win_reason = "max_turns_tiebreak"
            elif p1_props > p0_props:
                gs.winner = 1
                gs.win_reason = "max_turns_tiebreak"
            else:
                gs.winner = -1
                gs.win_reason = "max_turns_draw"
            return None

    opp_co_id = gs.co_states[opponent].co_id
    for unit in list(gs.units[opponent]):
        moved_previous_turn = unit.moved
        unit.moved = False
        stats = UNIT_STATS[unit.unit_type]
        drain = idle_start_of_day_fuel_drain(unit, opp_co_id)
        if moved_previous_turn and drain > 0:
            drain = 0
        unit.fuel = max(0, unit.fuel - drain)
        if unit.fuel == 0 and stats.unit_class in ("air", "copter", "naval"):
            prop = gs.get_property_at(*unit.pos)
            refuel_exempt = (
                prop is not None
                and prop.owner == opponent
                and (
                    (stats.unit_class == "naval" and prop.is_port)
                    or (stats.unit_class in ("air", "copter") and prop.is_airport)
                )
            )
            if not refuel_exempt:
                unit.hp = 0

    gs.units[opponent] = [u for u in gs.units[opponent] if u.is_alive]
    return opponent


def _make_patched_end_turn(
    ctx: dict[str, Any],
) -> tuple[Callable[[GameState], None], Callable[[GameState], None]]:
    """Return ``(patched, original_unbound)`` for ``GameState._end_turn``."""
    from engine import game as game_module

    original = game_module.GameState._end_turn

    def patched(self: GameState) -> None:
        opp = _run_end_turn_prefix_to_property_resupply(self)
        if opp is None:
            return

        env_i = ctx.get("current_env_i")
        if env_i is None:
            raise RuntimeError("probe ctx missing current_env_i")

        base = copy.deepcopy(self)
        s_ibr = copy.deepcopy(base)
        s_ibr._grant_income(opp)
        s_ibr._resupply_on_properties(opp)
        f_ibr = int(s_ibr.funds[opp])

        s_rbi = copy.deepcopy(base)
        s_rbi._resupply_on_properties(opp)
        s_rbi._grant_income(opp)
        f_rbi = int(s_rbi.funds[opp])

        frame_after = ctx.get("frame_after")
        # Tight exports omit ``frames[step_i+1]`` after the final envelope — no Tier-1
        # line to compare (``replay_snapshot_compare.replay_snapshot_pairing``).
        if frame_after is not None:
            awbw_map = ctx["awbw_to_engine"]
            php_all = _frame_funds_by_engine_seat(frame_after, awbw_map)
            php_opp = int(php_all[opp])

            ambiguous = f_ibr == f_rbi
            if ambiguous:
                binn = "AMBIGUOUS"
            elif php_opp == f_ibr and php_opp != f_rbi:
                binn = "PHP_MATCHES_IBR"
            elif php_opp == f_rbi and php_opp != f_ibr:
                binn = "PHP_MATCHES_RBI"
            else:
                binn = "PHP_MATCHES_NEITHER"

            ctx["records"].append(
                {
                    "env_i": env_i,
                    "turn_starter_seat": opp,
                    "php_funds_turn_starter": php_opp,
                    "engine_funds_ibr": f_ibr,
                    "engine_funds_rbi": f_rbi,
                    "bin": binn,
                }
            )

        self._resupply_on_properties(opp)
        self._grant_income(opp)
        self._refresh_comm_towers()

    return patched, original


def probe_one_game(gid: int) -> dict[str, Any]:
    from engine import game as game_module

    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id = {
        int(g["games_id"]): g
        for g in games.values()
        if isinstance(g, dict) and "games_id" in g
    }
    meta = by_id[gid]

    random.seed(_seed_for_game(CANONICAL_SEED, gid))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
    zpath = ZIPS / f"{gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    if replay_snapshot_pairing(len(frames), len(envs)) is None:
        return {
            "gid": gid,
            "co_p0": co0,
            "co_p1": co1,
            "matchup": meta.get("matchup"),
            "result": "unsupported_pairing",
            "n_frames": len(frames),
            "n_envelopes": len(envs),
            "records": [],
        }
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    ctx: dict[str, Any] = {
        "awbw_to_engine": awbw_to_engine,
        "records": [],
        "current_env_i": None,
        "frame_after": None,
    }
    patched, original = _make_patched_end_turn(ctx)

    game_module.GameState._end_turn = patched  # type: ignore[assignment]

    try:
        state = make_initial_state(
            map_data,
            co0,
            co1,
            starting_funds=0,
            tier_name=str(meta.get("tier") or "T2"),
            replay_first_mover=first_mover,
        )

        for env_i, (pid, day, actions) in enumerate(envs):
            ctx["current_env_i"] = env_i
            snap_i = env_i + 1
            ctx["frame_after"] = frames[snap_i] if snap_i < len(frames) else None
            try:
                for obj in actions:
                    apply_oracle_action_json(
                        state,
                        obj,
                        awbw_to_engine,
                        envelope_awbw_player_id=pid,
                    )
            except UnsupportedOracleAction as e:
                return {
                    "gid": gid,
                    "co_p0": co0,
                    "co_p1": co1,
                    "matchup": meta.get("matchup"),
                    "result": "oracle_gap",
                    "message": str(e),
                    "records": list(ctx["records"]),
                }
            except Exception as e:
                return {
                    "gid": gid,
                    "result": "fatal",
                    "message": f"{type(e).__name__}: {e}",
                    "records": list(ctx["records"]),
                }

        return {
            "gid": gid,
            "co_p0": co0,
            "co_p1": co1,
            "matchup": meta.get("matchup"),
            "result": "completed",
            "records": list(ctx["records"]),
        }
    finally:
        game_module.GameState._end_turn = original  # type: ignore[assignment]


def _pick_sample_gids(
    register_path: Path,
    n: int,
    prefer_ams_only: bool,
) -> list[int]:
    rows = [
        json.loads(ln)
        for ln in register_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    ok = [r for r in rows if r.get("class") == "ok"]
    out: list[int] = []

    def eligible(r: dict) -> bool:
        return (
            r["co_p0_id"] not in EXCLUDED_CO_IDS
            and r["co_p1_id"] not in EXCLUDED_CO_IDS
        )

    if prefer_ams_only:
        ams = [
            int(r["games_id"])
            for r in ok
            if eligible(r)
            and r["co_p0_id"] in (1, 7, 8)
            and r["co_p1_id"] in (1, 7, 8)
        ]
        ams.sort()
        out.extend(ams[:n])

    if len(out) < n:
        rest = [
            int(r["games_id"])
            for r in ok
            if eligible(r)
            and int(r["games_id"]) not in out
        ]
        rest.sort()
        for g in rest:
            if len(out) >= n:
                break
            out.append(g)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register", type=Path, default=ROOT / "logs" / "desync_register_post_phase11j_combined.jsonl")
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--no-prefer-ams", action="store_true", help="Do not prefer Andy/Max/Sami mirror matchups first.")
    ap.add_argument("--gid", type=int, action="append", help="Explicit gid(s); overrides sampling.")
    ap.add_argument("--out-json", type=Path, default=ROOT / "logs" / "phase11j_funds_ordering_probe.json")
    args = ap.parse_args()

    if args.gid:
        gids = args.gid
    else:
        gids = _pick_sample_gids(
            args.register,
            args.sample_size,
            prefer_ams_only=not args.no_prefer_ams,
        )

    cases: list[dict[str, Any]] = []
    global_bins: Counter[str] = Counter()
    for gid in gids:
        try:
            cases.append(probe_one_game(gid))
        except Exception as e:
            cases.append({"gid": gid, "result": "probe_exception", "message": f"{type(e).__name__}: {e}"})

    for c in cases:
        if c.get("result") != "completed":
            continue
        for r in c.get("records") or []:
            global_bins[r["bin"]] += 1

    summary = {
        "n_games_requested": len(gids),
        "gids": gids,
        "aggregate_bins": dict(global_bins),
        "per_game_bins": {},
    }
    for c in cases:
        gid = c["gid"]
        bc = Counter((r.get("bin") or "?") for r in (c.get("records") or []))
        summary["per_game_bins"][gid] = dict(bc)

    out_obj = {"summary": summary, "cases": cases}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(out_obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["aggregate_bins"], indent=2))
    incomplete = [c for c in cases if c.get("result") != "completed"]
    if incomplete:
        print(f"\nNon-completed: {len(incomplete)} (see cases in JSON)")
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
