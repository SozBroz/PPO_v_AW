#!/usr/bin/env python3
"""For one gid + transition, dump per-unit HP delta and inferred PHP repair cost.

We want to know: at the start-of-Rachel-turn repair tick (which fires at the
end of Drake's turn in this engine), what are the per-unit costs PHP charged
vs what the engine charged, and are there any units that PHP healed but
engine didn't (or vice versa)?
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state, _property_day_repair_gold
from engine.unit import UNIT_STATS, UnitType
from engine.map_loader import load_map
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

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
CATALOG_EXTRAS = ROOT / "data" / "amarriner_gl_extras_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


_NAME_TO_UT = {
    "Infantry": UnitType.INFANTRY, "Mech": UnitType.MECH,
    "Recon": UnitType.RECON, "APC": UnitType.APC,
    "Tank": UnitType.TANK, "Md.Tank": UnitType.MED_TANK,
    "Neotank": UnitType.NEO_TANK, "Megatank": UnitType.MEGA_TANK,
    "Anti-Air": UnitType.ANTI_AIR, "Artillery": UnitType.ARTILLERY,
    "Rockets": UnitType.ROCKET, "Missiles": UnitType.MISSILES,
    "Piperunner": UnitType.PIPERUNNER,
    "Battle Copter": UnitType.B_COPTER, "Transport Copter": UnitType.T_COPTER,
    "Fighter": UnitType.FIGHTER, "Bomber": UnitType.BOMBER,
    "Stealth": UnitType.STEALTH, "Black Bomb": UnitType.BLACK_BOMB,
    "Battleship": UnitType.BATTLESHIP, "Cruiser": UnitType.CRUISER,
    "Lander": UnitType.LANDER, "Sub": UnitType.SUBMARINE,
    "Black Boat": UnitType.BLACK_BOAT, "Carrier": UnitType.CARRIER,
}


def _php_units_at(frame: dict, awbw_to_engine: dict[int, int]) -> dict[tuple[int, int], dict]:
    out = {}
    for u in (frame.get("units") or {}).values():
        try:
            x = int(u.get("x"))
            y = int(u.get("y"))
            hp_internal = int(round(float(u.get("hit_points")) * 10))
            apid = int(u.get("players_id"))
        except (TypeError, ValueError):
            continue
        if apid not in awbw_to_engine:
            continue
        out[(y, x)] = {
            "name": u.get("name"),
            "hp": hp_internal,
            "player": awbw_to_engine[apid],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env-pre", type=int, required=True,
                    help="envelope index whose post-snapshot is BEFORE the tick")
    ap.add_argument("--env-post", type=int, required=True,
                    help="envelope index whose post-snapshot is AFTER the tick")
    ap.add_argument("--player", type=int, required=True,
                    help="engine player whose repair tick is being analyzed")
    args = ap.parse_args()

    by_id = {}
    for cat_path in (CATALOG, CATALOG_EXTRAS):
        if cat_path.exists():
            cat = json.loads(cat_path.read_text(encoding="utf-8"))
            for g in (cat.get("games") or {}).values():
                if isinstance(g, dict) and "games_id" in g:
                    by_id[int(g["games_id"])] = g
    meta = by_id[args.gid]

    random.seed(_seed_for_game(CANONICAL_SEED, args.gid))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)

    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    pre_frame = frames[args.env_pre + 1]
    post_frame = frames[args.env_post + 1]
    php_pre = _php_units_at(pre_frame, awbw_to_engine)
    php_post = _php_units_at(post_frame, awbw_to_engine)

    eng_pre = None
    eng_post = None
    eng_pre_funds = None
    eng_post_funds = None

    for env_i, (pid, day, actions) in enumerate(envs):
        try:
            for obj in actions:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
        except UnsupportedOracleAction as e:
            print(f"FAIL at env {env_i}: {e}")
            return 1

        if env_i == args.env_pre:
            eng_pre = {(u.pos): {"name": u.unit_type.name, "hp": u.hp,
                                  "ut": u.unit_type, "player": p}
                       for p in (0, 1) for u in state.units[p]}
            eng_pre_funds = (int(state.funds[0]), int(state.funds[1]))

        if env_i == args.env_post:
            eng_post = {(u.pos): {"name": u.unit_type.name, "hp": u.hp,
                                   "ut": u.unit_type, "player": p}
                        for p in (0, 1) for u in state.units[p]}
            eng_post_funds = (int(state.funds[0]), int(state.funds[1]))
            break

    print(f"=== gid {args.gid} player {args.player} env {args.env_pre} -> {args.env_post} ===")
    print(f"  engine funds pre={eng_pre_funds} post={eng_post_funds}")
    print(f"  PHP    funds pre/post: see drill")

    eng_player_pre = {pos: u for pos, u in eng_pre.items() if u["player"] == args.player}
    eng_player_post = {pos: u for pos, u in eng_post.items() if u["player"] == args.player}
    php_player_pre = {pos: u for pos, u in php_pre.items() if u["player"] == args.player}
    php_player_post = {pos: u for pos, u in php_post.items() if u["player"] == args.player}

    eng_repair_total = 0
    php_repair_total = 0
    rows = []
    all_pos = sorted(set(eng_player_pre.keys()) | set(eng_player_post.keys())
                     | set(php_player_pre.keys()) | set(php_player_post.keys()))
    for pos in all_pos:
        e_pre = eng_player_pre.get(pos)
        e_post = eng_player_post.get(pos)
        p_pre = php_player_pre.get(pos)
        p_post = php_player_post.get(pos)

        # Only consider positions that exist in both pre+post for at least one side
        # and where the unit type matches across pre/post
        e_heal = 0
        e_cost = 0
        if e_pre and e_post and e_pre.get("name") == e_post.get("name"):
            e_heal = max(0, e_post["hp"] - e_pre["hp"])
            if e_heal > 0:
                e_cost = _property_day_repair_gold(e_heal, e_pre["ut"])
        p_heal = 0
        p_cost = 0
        if p_pre and p_post and p_pre.get("name") == p_post.get("name"):
            p_heal = max(0, p_post["hp"] - p_pre["hp"])
            if p_heal > 0:
                # find matching unit type from name
                ut = _NAME_TO_UT.get(p_pre["name"])
                if ut is not None:
                    p_cost = _property_day_repair_gold(p_heal, ut)

        if e_heal > 0 or p_heal > 0:
            eng_repair_total += e_cost
            php_repair_total += p_cost
            rows.append({
                "pos": pos,
                "name": (e_pre or p_pre).get("name") if (e_pre or p_pre) else "?",
                "eng_pre": e_pre["hp"] if e_pre else None,
                "eng_post": e_post["hp"] if e_post else None,
                "eng_heal": e_heal, "eng_cost": e_cost,
                "php_pre": p_pre["hp"] if p_pre else None,
                "php_post": p_post["hp"] if p_post else None,
                "php_heal": p_heal, "php_cost": p_cost,
                "diff_cost": e_cost - p_cost,
            })

    print(f"\n  per-unit repair (player {args.player}):")
    for r in rows:
        marker = " <<" if r["diff_cost"] != 0 else ""
        print(f"    {r['pos']} {r['name']:14s}  eng:{r['eng_pre']}->{r['eng_post']}(+{r['eng_heal']}) ${r['eng_cost']:5d} | "
              f"php:{r['php_pre']}->{r['php_post']}(+{r['php_heal']}) ${r['php_cost']:5d} | dcost={r['diff_cost']:+5d}{marker}")
    print(f"\n  TOTALS — engine_repair=${eng_repair_total}, php_repair=${php_repair_total}, diff={eng_repair_total - php_repair_total:+d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
