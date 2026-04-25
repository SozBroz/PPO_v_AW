#!/usr/bin/env python3
"""Phase 11J-FINAL — Compare engine repair pass vs PHP unit HP at envelope boundary.

For each envelope from --from-env onward:
  1. Apply envelope actions in engine.
  2. Snapshot engine units (player, pos, unit_type, hp).
  3. Read PHP frame for the same envelope index.
  4. Print diffs.

Use to determine why a repair-pass engine charge differs from PHP.
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

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, UnitType
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
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


_PHP_UNIT_TYPE_TO_NAME = {
    "Infantry": "INFANTRY", "Mech": "MECH",
    "Recon": "RECON", "Tank": "TANK", "Md.Tank": "MD_TANK",
    "Neotank": "NEO_TANK", "Megatank": "MEGA_TANK",
    "APC": "APC", "Artillery": "ARTILLERY", "Rockets": "ROCKETS",
    "Missiles": "MISSILES", "Anti-Air": "ANTI_AIR",
    "Battle Copter": "B_COPTER", "Transport Copter": "T_COPTER",
    "Fighter": "FIGHTER", "Bomber": "BOMBER", "Stealth": "STEALTH",
    "Black Bomb": "BLACK_BOMB", "Battleship": "BATTLESHIP",
    "Cruiser": "CRUISER", "Lander": "LANDER", "Sub": "SUB",
    "Black Boat": "BLACK_BOAT", "Carrier": "CARRIER",
    "Piperunner": "PIPERUNNER",
}


def php_units_for_player(frame: dict, awbw_pid: int) -> list[dict]:
    out = []
    for u in (frame.get("units") or {}).values():
        try:
            pid = int(u.get("players_id") or u.get("units_players_id") or 0)
        except (TypeError, ValueError):
            continue
        if pid != awbw_pid:
            continue
        try:
            x = int(u.get("x") if u.get("x") is not None else u.get("units_x"))
            y = int(u.get("y") if u.get("y") is not None else u.get("units_y"))
            hp_raw = u.get("hit_points")
            if hp_raw is None:
                hp_raw = u.get("units_hit_points") or 10
            hp_int = int(round(float(hp_raw) * 10))
        except (TypeError, ValueError):
            continue
        out.append({
            "name": u.get("name") or u.get("units_name"),
            "x": x, "y": y,
            "hp_int": hp_int,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--from-env", type=int, required=True)
    ap.add_argument("--to-env", type=int, default=None)
    ap.add_argument("--player", type=int, choices=[0, 1], default=None,
                    help="Engine player to focus diff on (default: both).")
    args = ap.parse_args()

    by_id = {}
    for cat_path in (CATALOG, CATALOG_EXTRAS):
        if not cat_path.exists():
            continue
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
    engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )
    to_env = args.to_env if args.to_env is not None else len(envs)

    for env_i, (pid, day, actions) in enumerate(envs):
        try:
            for obj in actions:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    envelope_awbw_player_id=pid,
                )
        except Exception as e:
            print(f"FAIL env {env_i}: {type(e).__name__}: {e}")
            break
        if env_i + 1 >= len(frames):
            continue
        if env_i < args.from_env or env_i >= to_env:
            continue
        frame = frames[env_i + 1]
        seats = (0, 1) if args.player is None else (args.player,)
        print(f"\n=== POST-ENV {env_i} (acting pid={pid} day={day}) ===")
        for seat in seats:
            php_pid = engine_to_awbw[seat]
            eng_units = []
            for u in state.units[seat]:
                eng_units.append((u.unit_type.name, (u.pos[1], u.pos[0]), u.hp))
            php_units = php_units_for_player(frame, php_pid)
            php_set = {(p["name"], p["x"], p["y"]): p["hp_int"] for p in php_units}
            eng_dict = {(name, pos[0], pos[1]): hp for name, pos, hp in eng_units}
            print(f"  P{seat} (awbw={php_pid}): "
                  f"engine_funds={state.funds[seat]} "
                  f"php_funds={[int(pl.get('funds') or 0) for pl in (frame.get('players') or {}).values() if int(pl.get('id') or 0) == php_pid][0]}")
            php_only = []
            eng_only = []
            mismatched = []
            for key, hp in eng_dict.items():
                php_name = next((k for k in php_set if k[1] == key[1] and k[2] == key[2]), None)
                if php_name is None:
                    eng_only.append((key, hp))
                else:
                    if php_set[php_name] != hp or php_name[0] not in (key[0], _PHP_UNIT_TYPE_TO_NAME.get(key[0])):
                        if php_set[php_name] != hp:
                            mismatched.append((key, hp, php_name, php_set[php_name]))
            for php_name, php_hp in php_set.items():
                eng_name = next((k for k in eng_dict if k[1] == php_name[1] and k[2] == php_name[2]), None)
                if eng_name is None:
                    php_only.append((php_name, php_hp))
            if mismatched:
                print(f"    HP MISMATCH ({len(mismatched)}):")
                for k, ehp, pname, php_hp in mismatched:
                    print(f"      {k} engine_hp={ehp} php_name={pname[0]} php_hp_int={php_hp}")
            if php_only:
                print(f"    PHP-only units: {php_only}")
            if eng_only:
                print(f"    ENGINE-only units: {eng_only}")
            if not (mismatched or php_only or eng_only):
                print(f"    units MATCH ({len(eng_dict)} units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
