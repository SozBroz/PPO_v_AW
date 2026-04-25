#!/usr/bin/env python3
"""Phase 11J-FUNDS-DEEP — per-unit eligibility drill on PHP_MATCHES_NEITHER rows.

Reads ``logs/phase11j_funds_ordering_probe.json``, replays each NEITHER
gid up to the offending envelope, and at the resupply boundary captures:
- engine units list for the turn-starter (hp, pos, cls, eligible? cost-if-healed)
- PHP units list for the same player at frame[snap_i]
- engine vs PHP per-unit hp (paired by position+type)
- for each side: total funds spent on day-property repair if applied as full step

Also tracks loaded units (cargo) to test the "loaded units inside transports
on properties" hypothesis.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import ActionStage
from engine.game import GameState, _property_day_repair_gold, make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType, idle_start_of_day_fuel_drain
from engine.terrain import get_terrain
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
from tools._phase11j_funds_ordering_probe import (
    _run_end_turn_prefix_to_property_resupply,
)

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


# Map PHP unit name → engine UnitType
PHP_NAME_TO_UNIT_TYPE = {
    "Infantry": UnitType.INFANTRY,
    "Mech": UnitType.MECH,
    "Recon": UnitType.RECON,
    "Tank": UnitType.TANK,
    "Md.Tank": UnitType.MED_TANK,
    "Md. Tank": UnitType.MED_TANK,
    "Medium Tank": UnitType.MED_TANK,
    "Neotank": UnitType.NEO_TANK,
    "Neo Tank": UnitType.NEO_TANK,
    "Megatank": UnitType.MEGA_TANK,
    "Mega Tank": UnitType.MEGA_TANK,
    "APC": UnitType.APC,
    "Artillery": UnitType.ARTILLERY,
    "Rocket": UnitType.ROCKET,
    "Rockets": UnitType.ROCKET,
    "Anti-Air": UnitType.ANTI_AIR,
    "Anti Air": UnitType.ANTI_AIR,
    "Missile": UnitType.MISSILES,
    "Missiles": UnitType.MISSILES,
    "Piperunner": UnitType.PIPERUNNER,
    "Pipe Runner": UnitType.PIPERUNNER,
    "T-Copter": UnitType.T_COPTER,
    "B-Copter": UnitType.B_COPTER,
    "Fighter": UnitType.FIGHTER,
    "Bomber": UnitType.BOMBER,
    "Stealth": UnitType.STEALTH,
    "B-Bomber": UnitType.BLACK_BOMB,
    "Black Bomb": UnitType.BLACK_BOMB,
    "Lander": UnitType.LANDER,
    "Cruiser": UnitType.CRUISER,
    "Sub": UnitType.SUBMARINE,
    "Submarine": UnitType.SUBMARINE,
    "Battleship": UnitType.BATTLESHIP,
    "B-Ship": UnitType.BATTLESHIP,
    "Carrier": UnitType.CARRIER,
    "B-Boat": UnitType.BLACK_BOAT,
    "Black Boat": UnitType.BLACK_BOAT,
}


def _get_unit_class(ut):
    return UNIT_STATS[ut].unit_class


def _qualifies_heal_for_prop(prop, ut: UnitType) -> bool:
    if prop is None:
        return False
    if prop.is_lab or prop.is_comm_tower:
        return False
    cls = _get_unit_class(ut)
    is_city = not (
        prop.is_hq or prop.is_lab or prop.is_comm_tower
        or prop.is_base or prop.is_airport or prop.is_port
    )
    if cls in ("infantry", "mech", "vehicle", "pipe"):
        return prop.is_hq or prop.is_base or is_city
    if cls in ("air", "copter"):
        return prop.is_airport
    if cls == "naval":
        return prop.is_port
    return False


def _php_owned_prop_at(frame: dict, owner_country_id: int, y: int, x: int):
    """Return a TerrainInfo for the property at (y,x) IF owned by ``owner_country_id``.

    PHP frames encode ownership via terrain_id (each country has its own
    terrain id range). owner_country_id is the AWBW countries_id (1=OS,
    2=BM, 4=YC, etc.).
    """
    for b in (frame.get("buildings") or {}).values():
        try:
            by = int(b.get("y"))
            bx = int(b.get("x"))
        except (TypeError, ValueError):
            continue
        if by != y or bx != x:
            continue
        tid_raw = b.get("terrain_id") or b.get("buildings_terrain_id") or b.get("type")
        try:
            tid = int(tid_raw)
        except (TypeError, ValueError):
            return None
        info = get_terrain(tid)
        if info is None or not info.is_property:
            return None
        if info.country_id != owner_country_id:
            return None
        return info
    return None


def _php_qualifies(info, ut: UnitType) -> bool:
    if info is None:
        return False
    if info.is_lab or info.is_comm_tower:
        return False
    cls = _get_unit_class(ut)
    is_city = not (
        info.is_hq or info.is_lab or info.is_comm_tower
        or info.is_base or info.is_airport or info.is_port
    )
    if cls in ("infantry", "mech", "vehicle", "pipe"):
        return info.is_hq or info.is_base or is_city
    if cls in ("air", "copter"):
        return info.is_airport
    if cls == "naval":
        return info.is_port
    return False


def _engine_eligible(state: GameState, player: int) -> list[dict]:
    out = []
    for u in state.units[player]:
        prop = state.get_property_at(*u.pos)
        if prop is None or prop.owner != player:
            continue
        if not _qualifies_heal_for_prop(prop, u.unit_type):
            continue
        if u.hp >= 100:
            continue
        co_id = state.co_states[player].co_id
        property_heal = 30 if co_id == 28 else 20
        desired = min(property_heal, 100 - u.hp)
        cost_full = _property_day_repair_gold(desired, u.unit_type)
        out.append({
            "type": u.unit_type.name, "pos": list(u.pos), "hp": int(u.hp),
            "cost_if_full": int(cost_full), "desired_internal": int(desired),
            "loaded": [c.unit_type.name for c in u.loaded_units] if u.loaded_units else [],
        })
    return out


def _php_eligible(frame: dict, awbw_pid: int, owner_country_id: int, co_id: int) -> list[dict]:
    """Mimic engine eligibility but on PHP frame; PHP hit_points is float 0-10."""
    out = []
    units = frame.get("units") or {}
    for uid, pu in units.items():
        try:
            pl_id = int(pu.get("players_id"))
        except (TypeError, ValueError):
            continue
        if pl_id != awbw_pid:
            continue
        try:
            y = int(pu.get("y"))
            x = int(pu.get("x"))
        except (TypeError, ValueError):
            continue
        carried = (str(pu.get("carried") or "")).strip().upper() == "Y"
        name = (pu.get("name") or "").strip()
        ut = PHP_NAME_TO_UNIT_TYPE.get(name)
        if ut is None:
            continue
        info = _php_owned_prop_at(frame, owner_country_id, y, x)
        if not _php_qualifies(info, ut):
            continue
        try:
            hp_internal_disp = float(pu.get("hit_points") or 0)
        except (TypeError, ValueError):
            hp_internal_disp = 0
        # PHP hp is 0-10 float; convert to engine 0-100 by *10
        hp_eng = int(round(hp_internal_disp * 10))
        if hp_eng >= 100:
            continue
        property_heal = 30 if co_id == 28 else 20
        desired = min(property_heal, 100 - hp_eng)
        cost_full = _property_day_repair_gold(desired, ut)
        out.append({
            "type": ut.name, "pos": [y, x], "hp_php": hp_internal_disp,
            "hp_eng": hp_eng, "cost_if_full": int(cost_full),
            "carried": carried,
            "uid": int(pu.get("id") or 0),
        })
    return out


def deep_drill(gid: int, target_envs: list[int]) -> dict:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id = {int(g["games_id"]): g for g in games.values()
             if isinstance(g, dict) and "games_id" in g}
    meta = by_id[gid]

    random.seed(_seed_for_game(CANONICAL_SEED, gid))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
    zpath = ZIPS / f"{gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    if replay_snapshot_pairing(len(frames), len(envs)) is None:
        return {"gid": gid, "result": "unsupported_pairing"}
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    # Derive AWBW player_id -> terrain country_id mapping by scanning frames
    # for units standing on buildings (capture-completed units). The terrain
    # country_id encoded in PHP buildings differs from `players[].countries_id`
    # (e.g. PC=12 in terrain art but countries_id=17 in player record).
    awbw_pid_to_terrain_country: dict[int, int] = {}
    all_terrain_countries: set[int] = set()
    for f in frames:
        for b in (f.get("buildings") or {}).values():
            try:
                tid = int(b.get("terrain_id"))
            except (TypeError, ValueError):
                continue
            info = get_terrain(tid)
            if info is not None and info.country_id is not None:
                all_terrain_countries.add(int(info.country_id))
        bld_by_pos = {(int(b.get("y", -1)), int(b.get("x", -1))): b
                      for b in (f.get("buildings") or {}).values()}
        for u in (f.get("units") or {}).values():
            try:
                pos = (int(u.get("y")), int(u.get("x")))
                pid_awbw = int(u.get("players_id"))
            except (TypeError, ValueError):
                continue
            b = bld_by_pos.get(pos)
            if b is None:
                continue
            try:
                tid = int(b.get("terrain_id"))
            except (TypeError, ValueError):
                continue
            info = get_terrain(tid)
            if info is None or info.country_id is None:
                continue
            awbw_pid_to_terrain_country.setdefault(pid_awbw, int(info.country_id))
        if (len(awbw_pid_to_terrain_country) >= len(awbw_to_engine)
                and len(awbw_pid_to_terrain_country) >= 2):
            break
    # If only one mapping found and exactly two terrain countries exist,
    # infer the other player by elimination.
    if (len(awbw_pid_to_terrain_country) == 1
            and len(all_terrain_countries) == 2
            and len(awbw_to_engine) == 2):
        known_pid, known_country = next(iter(awbw_pid_to_terrain_country.items()))
        other_country = next(c for c in all_terrain_countries if c != known_country)
        other_pid = next(p for p in awbw_to_engine if p != known_pid)
        awbw_pid_to_terrain_country[other_pid] = other_country

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    target_set = set(target_envs)
    results = []

    for env_i, (pid, day, actions) in enumerate(envs):
        captured_pre_end = None
        try:
            for obj in actions:
                kind = obj.get("action") if isinstance(obj, dict) else None
                if env_i in target_set and kind == "End" and captured_pre_end is None:
                    # Snapshot state right before End triggers _end_turn.
                    captured_pre_end = copy.deepcopy(state)
                apply_oracle_action_json(state, obj, awbw_to_engine,
                                         envelope_awbw_player_id=pid)
        except UnsupportedOracleAction as e:
            return {"gid": gid, "result": f"oracle_gap@{env_i}", "err": str(e)}

        if env_i not in target_set:
            continue

        if captured_pre_end is None:
            # No explicit End action in this envelope — fall back to current state
            results.append({"env_i": env_i, "skipped": "no_end_action_seen"})
            continue

        # Run the prefix (fuel/crash + active_player switch) on the snapshot to
        # match probe semantics, but NOT the resupply/income.
        deep = captured_pre_end
        opp = _run_end_turn_prefix_to_property_resupply(deep)
        if opp is None:
            results.append({"env_i": env_i, "skipped": "max_turns"})
            continue

        snap_i = env_i + 1
        frame_after = frames[snap_i] if snap_i < len(frames) else None
        frame_before = frames[env_i] if env_i < len(frames) else None
        if frame_after is None:
            results.append({"env_i": env_i, "skipped": "no_frame_after"})
            continue
        opp_awbw_pid = engine_to_awbw[opp]
        opp_co_id = deep.co_states[opp].co_id
        opp_country = awbw_pid_to_terrain_country.get(opp_awbw_pid)

        engine_elig = _engine_eligible(deep, opp)
        php_elig = (_php_eligible(frame_after, opp_awbw_pid, opp_country, opp_co_id)
                    if opp_country is not None else [])

        eng_total_full = sum(u["cost_if_full"] for u in engine_elig)
        php_total_full = sum(u["cost_if_full"] for u in php_elig)

        # Compare unit-by-unit by (type, pos)
        eng_by_pos = {(u["type"], tuple(u["pos"])): u for u in engine_elig}
        php_by_pos = {(u["type"], tuple(u["pos"])): u for u in php_elig}

        only_php = [u for k, u in php_by_pos.items() if k not in eng_by_pos]
        only_eng = [u for k, u in eng_by_pos.items() if k not in php_by_pos]
        common = [(eng_by_pos[k], php_by_pos[k]) for k in eng_by_pos.keys() if k in php_by_pos]

        # Compute hypothetical IBR / RBI on a fresh copy and the actual engine-style
        # repair pass to see what engine actually spends.
        n_income_props = deep.count_income_properties(opp)
        income = n_income_props * 1000
        if opp_co_id in (15, 19):
            income += n_income_props * 100
        funds_pre_pass = int(deep.funds[opp])

        ibr = copy.deepcopy(deep)
        ibr._grant_income(opp)
        ibr._resupply_on_properties(opp)
        ibr_funds = int(ibr.funds[opp])
        ibr_repair_spend = funds_pre_pass + income - ibr_funds

        def _opp_funds(frame):
            if frame is None: return None
            for pl in (frame.get("players") or {}).values():
                try:
                    if int(pl.get("id")) == opp_awbw_pid:
                        return int(pl.get("funds") or 0)
                except (TypeError, ValueError):
                    pass
            return None
        php_funds = _opp_funds(frame_after)
        php_funds_before = _opp_funds(frame_before)
        php_repair_spend = (
            (php_funds_before + income - php_funds)
            if (php_funds is not None and php_funds_before is not None)
            else None
        )

        # Compare per-unit cost on common units
        common_diffs = []
        for eu, pu in common:
            if eu["cost_if_full"] != pu["cost_if_full"]:
                common_diffs.append({
                    "type": eu["type"], "pos": eu["pos"],
                    "engine_hp": eu["hp"], "php_hp": pu["hp_php"],
                    "engine_cost": eu["cost_if_full"], "php_cost": pu["cost_if_full"],
                })

        results.append({
            "env_i": env_i,
            "day": day,
            "opp_engine_seat": opp,
            "opp_awbw_pid": opp_awbw_pid,
            "opp_country_id": opp_country,
            "opp_co_id": opp_co_id,
            "n_income_props": n_income_props,
            "income": income,
            "funds_pre_pass": funds_pre_pass,
            "engine_eligible_count": len(engine_elig),
            "php_eligible_count": len(php_elig),
            "engine_total_cost_if_full": eng_total_full,
            "php_total_cost_if_full": php_total_full,
            "only_php_eligible": only_php,
            "only_engine_eligible": only_eng,
            "common_count": len(common),
            "common_cost_diffs": common_diffs,
            "engine_ibr_funds": ibr_funds,
            "engine_ibr_repair_spend": ibr_repair_spend,
            "php_funds": php_funds,
            "php_funds_before_env": php_funds_before,
            "php_repair_spend": php_repair_spend,
            "delta_php_minus_engine_spend": (
                (php_repair_spend - ibr_repair_spend)
                if php_repair_spend is not None else None
            ),
        })

        # Continue the env loop normally — apply_oracle_action_json above
        # handled the actual engine end_turn via the 'End' action embedded
        # in the envelope. Don't double-step state.

    return {
        "gid": gid,
        "co_p0": co0, "co_p1": co1,
        "matchup": meta.get("matchup"),
        "result": "completed",
        "rows": results,
    }


def _state_pre_funds(state: GameState, env_i: int, opp: int) -> int:
    return int(state.funds[opp])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-json", type=Path,
                    default=ROOT / "logs" / "phase11j_funds_ordering_probe.json")
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "logs" / "phase11j_funds_deep_drill.json")
    ap.add_argument("--gid", type=int, action="append",
                    help="Restrict to specific gids (default: all NEITHER gids).")
    args = ap.parse_args()

    probe = json.loads(args.probe_json.read_text(encoding="utf-8"))
    targets: dict[int, list[int]] = defaultdict(list)
    for c in probe["cases"]:
        for r in c.get("records", []):
            if r.get("bin") == "PHP_MATCHES_NEITHER":
                gid = int(c["gid"])
                if args.gid and gid not in args.gid:
                    continue
                targets[gid].append(int(r["env_i"]))

    print(f"Drilling {sum(len(v) for v in targets.values())} NEITHER rows in "
          f"{len(targets)} gids")
    cases = []
    for gid, envs in sorted(targets.items()):
        try:
            cases.append(deep_drill(gid, envs))
        except Exception as e:
            cases.append({"gid": gid, "result": "exception",
                          "err": f"{type(e).__name__}: {e}"})
        print(f"  gid={gid}: {cases[-1].get('result')}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"cases": cases}, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
