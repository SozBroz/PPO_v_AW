#!/usr/bin/env python3
"""Dump engine vs PHP units for one player at a target envelope boundary."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
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


def _player_country_map(frames):
    out = {}
    for f in frames:
        bld_by_pos = {(int(b.get("y", -1)), int(b.get("x", -1))): b
                      for b in (f.get("buildings") or {}).values()}
        for u in (f.get("units") or {}).values():
            try:
                pos = (int(u["y"]), int(u["x"]))
                pid = int(u["players_id"])
            except (TypeError, ValueError, KeyError):
                continue
            b = bld_by_pos.get(pos)
            if not b:
                continue
            try:
                tid = int(b["terrain_id"])
            except (TypeError, ValueError, KeyError):
                continue
            info = get_terrain(tid)
            if info and info.country_id is not None:
                out.setdefault(pid, int(info.country_id))
        if len(out) >= 2:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env", type=int, required=True,
                    help="Envelope index where 'End' triggers turn-roll")
    args = ap.parse_args()

    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id = {int(g["games_id"]): g for g in games.values()
             if isinstance(g, dict) and "games_id" in g}
    meta = by_id[args.gid]

    random.seed(_seed_for_game(CANONICAL_SEED, args.gid))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    if replay_snapshot_pairing(len(frames), len(envs)) is None:
        print("unsupported pairing")
        return 1
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    pid_to_country = _player_country_map(frames)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    captured = None
    for env_i, (pid, day, actions) in enumerate(envs):
        for obj in actions:
            kind = obj.get("action") if isinstance(obj, dict) else None
            if env_i == args.env and kind == "End" and captured is None:
                captured = copy.deepcopy(state)
            apply_oracle_action_json(state, obj, awbw_to_engine,
                                     envelope_awbw_player_id=pid)
        if env_i == args.env:
            break

    if captured is None:
        print(f"No End action seen in env {args.env}")
        return 1

    deep = captured
    opp = _run_end_turn_prefix_to_property_resupply(deep)
    if opp is None:
        print("max turns")
        return 1
    opp_awbw = engine_to_awbw[opp]
    opp_country = pid_to_country.get(opp_awbw)
    print(f"env={args.env} opp_engine={opp} opp_awbw={opp_awbw} opp_country={opp_country}")

    # Engine units for opp
    print("\nENGINE units for opp:")
    for u in deep.units[opp]:
        prop = deep.get_property_at(*u.pos)
        elig_prop = (prop is not None and prop.owner == opp)
        print(f"  pos={u.pos} type={u.unit_type.name} hp={u.hp} fuel={u.fuel} ammo={u.ammo} "
              f"prop_tid={prop.terrain_id if prop else 'None'} owner={prop.owner if prop else 'None'} elig_prop={elig_prop}")

    # PHP units for opp at frame[env+1]
    fa = frames[args.env + 1]
    print(f"\nPHP units for awbw_pid={opp_awbw} at frame[{args.env+1}] day={fa.get('day')}:")
    bld_by_pos = {(int(b['y']), int(b['x'])): b for b in (fa.get('buildings') or {}).values()}
    for u in (fa.get('units') or {}).values():
        try:
            pl_id = int(u['players_id'])
        except (TypeError, ValueError, KeyError):
            continue
        if pl_id != opp_awbw:
            continue
        pos = (int(u['y']), int(u['x']))
        b = bld_by_pos.get(pos)
        owner_country = None
        prop_kind = None
        if b is not None:
            info = get_terrain(int(b['terrain_id']))
            if info:
                owner_country = info.country_id
                prop_kind = info.name
        print(f"  pos={pos} name={u['name']} hp={u['hit_points']} fuel={u.get('fuel')} ammo={u.get('ammo')} "
              f"carried={u.get('carried')} prop={prop_kind} prop_country={owner_country} "
              f"owned_by_opp={owner_country == opp_country}")

    # PHP units BEFORE the env (at frame[env])
    fb = frames[args.env]
    print(f"\nPHP units for awbw_pid={opp_awbw} at frame[{args.env}] day={fb.get('day')}:")
    bld_by_pos = {(int(b['y']), int(b['x'])): b for b in (fb.get('buildings') or {}).values()}
    for u in (fb.get('units') or {}).values():
        try:
            pl_id = int(u['players_id'])
        except (TypeError, ValueError, KeyError):
            continue
        if pl_id != opp_awbw:
            continue
        pos = (int(u['y']), int(u['x']))
        b = bld_by_pos.get(pos)
        owner_country = None
        prop_kind = None
        if b is not None:
            info = get_terrain(int(b['terrain_id']))
            if info:
                owner_country = info.country_id
                prop_kind = info.name
        print(f"  pos={pos} name={u['name']} hp={u['hit_points']} fuel={u.get('fuel')} ammo={u.get('ammo')} "
              f"carried={u.get('carried')} prop={prop_kind} prop_country={owner_country} "
              f"owned_by_opp={owner_country == opp_country}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
