#!/usr/bin/env python3
"""Phase 11K-REPAIR-BREAKDOWN — at the boundary frame N, dump every
Sturm-owned property in PHP vs engine, and compare unit HP and repair
eligibility. Used to find the +800/+1000 g/day repair cost mismatch in
gid 1635679 starting env 25 day 13.

Usage: python tools/_phase11k_repair_breakdown.py --gid 1635679 --env 25
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
from engine.terrain import get_terrain
from engine.unit import UnitType, UNIT_STATS
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.replay_snapshot_compare import replay_snapshot_pairing

CATS = [
    ROOT / "data" / "amarriner_gl_std_catalog.json",
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
    ROOT / "data" / "amarriner_gl_colin_batch.json",
]
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
    ap.add_argument("--gid", type=int, default=1635679)
    ap.add_argument("--env", type=int, required=True,
                    help="Compare PHP frame[env+1] vs frame[env]; engine state at end of env.")
    args = ap.parse_args()

    by_id: dict[int, dict] = {}
    for cp in CATS:
        if not cp.exists():
            continue
        cat = json.loads(cp.read_text(encoding="utf-8"))
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

    print(f"gid={args.gid} co_p0={co0} co_p1={co1}")
    print(f"awbw_to_engine={awbw_to_engine}")
    print(f"pid_to_country={pid_to_country}")
    print()

    # Apply all envs through args.env (inclusive).
    for env_i, (pid, day, actions) in enumerate(envs):
        if env_i + 1 < len(frames):
            pin: dict[int, int] = {}
            for u in (frames[env_i + 1].get("units") or {}).values():
                try:
                    uid = int(u["id"])
                    hp = float(u["hit_points"])
                except (TypeError, ValueError, KeyError):
                    continue
                pin[uid] = max(0, min(100, int(round(hp * 10))))
            state._oracle_post_envelope_units_by_id = pin
            def_hits: dict[int, int] = {}
            for obj in actions:
                if not isinstance(obj, dict):
                    continue
                if obj.get("action") not in ("Fire", "AttackSeam"):
                    continue
                ci = obj.get("combatInfo")
                if not isinstance(ci, dict):
                    continue
                d = ci.get("defender")
                if not isinstance(d, dict):
                    continue
                try:
                    d_uid = int(d.get("units_id"))
                except (TypeError, ValueError):
                    continue
                def_hits[d_uid] = def_hits.get(d_uid, 0) + 1
            state._oracle_post_envelope_multi_hit_defenders = {
                uid for uid, c in def_hits.items() if c > 1
            }
        for obj in actions:
            apply_oracle_action_json(state, obj, awbw_to_engine,
                                     envelope_awbw_player_id=pid)
        if env_i >= args.env:
            break

    # At end of env=args.env, the active player has rolled to the next.
    # We're interested in the SAME player's start-of-next-turn behavior.
    # PHP frame[args.env + 1] = end of envelope (post-end-turn cadence).
    # PHP frame[args.env + 2] = end of NEXT envelope.
    # Sturm's day-N+1 income+repair happens at the START of env that begins day-N+1.

    f_post = frames[args.env + 1] if args.env + 1 < len(frames) else None
    f_next = frames[args.env + 2] if args.env + 2 < len(frames) else None

    # Find Sturm's pid (engine seat 0).
    sturm_engine_seat = 0
    sturm_awbw_pid = engine_to_awbw[sturm_engine_seat]
    sturm_country = pid_to_country.get(sturm_awbw_pid)

    print(f"=== STURM = engine_seat={sturm_engine_seat}, awbw_pid={sturm_awbw_pid}, country_id={sturm_country} ===")
    print()

    # Engine: Sturm's units and their property situation.
    print(f"=== ENGINE state at end of env {args.env} ===")
    print(f"  active_player={state.active_player}")
    print(f"  funds[0]={state.funds[0]}  funds[1]={state.funds[1]}")
    eng_props_by_pos = {(p.row, p.col): p for p in state.properties}
    eng_sturm_props = [p for p in state.properties if p.owner == sturm_engine_seat]
    eng_sturm_units = state.units[sturm_engine_seat]
    print(f"  Sturm property count={len(eng_sturm_props)}")
    print(f"  Sturm unit count={len(eng_sturm_units)}")

    print()
    print("ENGINE Sturm units on owned properties (eligible for repair):")
    for u in sorted(eng_sturm_units, key=lambda u: u.pos):
        prop = eng_props_by_pos.get(u.pos)
        if prop is None or prop.owner != sturm_engine_seat:
            continue
        print(f"   {u.unit_type.name:>10} {u.pos} hp={u.hp:3d}  prop_terrain={prop.terrain_id}")

    print()
    print(f"=== PHP frame[{args.env + 1}] (post-Lash-day-N) — Sturm's start-of-day units ===")
    if f_post:
        bld_by_pos = {(int(b['y']), int(b['x'])): b for b in (f_post.get('buildings') or {}).values()}
        php_sturm_props_post = []
        for b in (f_post.get('buildings') or {}).values():
            try:
                bx, by = int(b['x']), int(b['y'])
                tid = int(b['terrain_id'])
            except (TypeError, ValueError, KeyError):
                continue
            info = get_terrain(tid)
            if info and info.country_id == sturm_country:
                php_sturm_props_post.append(((by, bx), tid))
        print(f"  PHP Sturm property count = {len(php_sturm_props_post)}")

        eng_owned = {(p.row, p.col): p.terrain_id for p in eng_sturm_props}
        php_owned = {pos: tid for pos, tid in php_sturm_props_post}
        only_eng = sorted(set(eng_owned) - set(php_owned))
        only_php = sorted(set(php_owned) - set(eng_owned))
        print(f"  Properties only in ENGINE Sturm: {only_eng}")
        for pos in only_eng:
            print(f"     pos={pos} terrain={eng_owned[pos]}")
        print(f"  Properties only in PHP Sturm: {only_php}")
        for pos in only_php:
            print(f"     pos={pos} terrain={php_owned[pos]}")
        php_sturm_units_post = []
        for u in (f_post.get('units') or {}).values():
            try:
                pl_id = int(u['players_id'])
            except (TypeError, ValueError, KeyError):
                continue
            if pl_id != sturm_awbw_pid:
                continue
            pos = (int(u['y']), int(u['x']))
            b = bld_by_pos.get(pos)
            on_owned_prop = False
            if b:
                info = get_terrain(int(b['terrain_id']))
                if info and info.country_id == sturm_country:
                    on_owned_prop = True
            php_sturm_units_post.append((u.get('name'), pos, u.get('hit_points'), on_owned_prop))
        print(f"  PHP Sturm units on owned props (pre-repair):")
        for name, pos, hp, on_prop in sorted(php_sturm_units_post):
            if on_prop:
                print(f"     {name:>10} {pos} hp={hp}")

    print()
    print(f"=== PHP frame[{args.env + 2}] (post-Sturm-day-N+1) — units after income+repair ===")
    if f_next:
        bld_by_pos = {(int(b['y']), int(b['x'])): b for b in (f_next.get('buildings') or {}).values()}
        php_sturm_units_next = {}
        for u in (f_next.get('units') or {}).values():
            try:
                pl_id = int(u['players_id'])
            except (TypeError, ValueError, KeyError):
                continue
            if pl_id != sturm_awbw_pid:
                continue
            pos = (int(u['y']), int(u['x']))
            b = bld_by_pos.get(pos)
            on_owned_prop = False
            terrain = None
            if b:
                terrain = int(b['terrain_id'])
                info = get_terrain(terrain)
                if info and info.country_id == sturm_country:
                    on_owned_prop = True
            php_sturm_units_next[pos] = (u.get('name'), u.get('hit_points'), on_owned_prop, terrain)

        print(f"  PHP Sturm units on owned props (post-repair):")
        for pos in sorted(php_sturm_units_next):
            name, hp, on_prop, terrain = php_sturm_units_next[pos]
            if on_prop:
                print(f"     {name:>10} {pos} hp={hp}  terrain={terrain}")

        # Compute per-unit repair cost (display HP delta × cost/10)
        if f_post:
            bld_by_pos_post = {(int(b['y']), int(b['x'])): b for b in (f_post.get('buildings') or {}).values()}
            pre = {}
            for u in (f_post.get('units') or {}).values():
                try:
                    pl_id = int(u['players_id'])
                except (TypeError, ValueError, KeyError):
                    continue
                if pl_id != sturm_awbw_pid:
                    continue
                pos = (int(u['y']), int(u['x']))
                pre[pos] = (u.get('name'), float(u.get('hit_points', 0)))
            print()
            print(f"  PHP per-unit repair (pre to post):")
            php_total = 0
            for pos in sorted(php_sturm_units_next):
                name, post_hp, on_prop, terrain = php_sturm_units_next[pos]
                if not on_prop:
                    continue
                pre_entry = pre.get(pos)
                if pre_entry is None:
                    continue
                pre_name, pre_hp = pre_entry
                try:
                    post_hpf = float(post_hp)
                except (TypeError, ValueError):
                    continue
                # Find UnitType
                ut = None
                for k, v in {
                    "Infantry": UnitType.INFANTRY, "Mech": UnitType.MECH,
                    "Recon": UnitType.RECON, "APC": UnitType.APC,
                    "Tank": UnitType.TANK, "Md.Tank": UnitType.MED_TANK,
                    "Md. Tank": UnitType.MED_TANK, "Neotank": UnitType.NEO_TANK,
                    "Mega Tank": UnitType.MEGA_TANK, "Anti-Air": UnitType.ANTI_AIR,
                    "Missile": UnitType.MISSILES, "Missiles": UnitType.MISSILES,
                    "Rocket": UnitType.ROCKET, "Rockets": UnitType.ROCKET,
                    "Artillery": UnitType.ARTILLERY, "T-Copter": UnitType.T_COPTER,
                    "B-Copter": UnitType.B_COPTER, "Fighter": UnitType.FIGHTER,
                    "Bomber": UnitType.BOMBER, "Stealth": UnitType.STEALTH,
                    "B-Ship": UnitType.BATTLESHIP, "Battleship": UnitType.BATTLESHIP,
                    "Cruiser": UnitType.CRUISER, "Lander": UnitType.LANDER,
                    "Sub": UnitType.SUBMARINE, "Carrier": UnitType.CARRIER,
                    "Black Boat": UnitType.BLACK_BOAT, "Black Bomb": UnitType.BLACK_BOMB,
                    "Piperunner": UnitType.PIPERUNNER, "Gunboat": UnitType.GUNBOAT,
                }.items():
                    if name == k:
                        ut = v; break
                if ut is None:
                    continue
                stats = UNIT_STATS[ut]
                # Display HP delta from PHP
                delta_disp = max(0.0, post_hpf - pre_hp)
                cost = int(stats.cost // 10 * round(delta_disp))
                php_total += cost
                marker = "  " if delta_disp == 0 else "++"
                print(f"   {marker} {name:>10} {pos} pre={pre_hp:.1f} post={post_hpf:.1f} delta={delta_disp:+.1f}  cost={cost}")
            print(f"  PHP total Sturm repair = {php_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
