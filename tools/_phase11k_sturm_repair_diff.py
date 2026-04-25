#!/usr/bin/env python3
"""Phase 11K-STURM-REPAIR-DIFF — compare engine vs PHP unit set on
properties at the start of every P0 turn for gid 1635679. Emits the
delta in eligible-for-repair units, which is the funds-drift smoking
gun documented in
``docs/oracle_exception_audit/phase11j_final_build_no_op_residuals.md``.
"""
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
from engine.unit import UnitType, UNIT_STATS
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

CATS = [
    ROOT / "data" / "amarriner_gl_std_catalog.json",
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
    ROOT / "data" / "amarriner_gl_colin_batch.json",
]
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


_PHP_NAME_TO_UT = {
    "Infantry": UnitType.INFANTRY,
    "Mech": UnitType.MECH,
    "Recon": UnitType.RECON,
    "APC": UnitType.APC,
    "Tank": UnitType.TANK,
    "Md.Tank": UnitType.MED_TANK,
    "Md. Tank": UnitType.MED_TANK,
    "Neotank": UnitType.NEO_TANK,
    "Mega Tank": UnitType.MEGA_TANK,
    "Anti-Air": UnitType.ANTI_AIR,
    "Missile": UnitType.MISSILES,
    "Missiles": UnitType.MISSILES,
    "Rocket": UnitType.ROCKET,
    "Rockets": UnitType.ROCKET,
    "Artillery": UnitType.ARTILLERY,
    "T-Copter": UnitType.T_COPTER,
    "B-Copter": UnitType.B_COPTER,
    "Fighter": UnitType.FIGHTER,
    "Bomber": UnitType.BOMBER,
    "Stealth": UnitType.STEALTH,
    "B-Ship": UnitType.BATTLESHIP,
    "Battleship": UnitType.BATTLESHIP,
    "Cruiser": UnitType.CRUISER,
    "Lander": UnitType.LANDER,
    "Sub": UnitType.SUBMARINE,
    "Submarine": UnitType.SUBMARINE,
    "Carrier": UnitType.CARRIER,
    "Black Boat": UnitType.BLACK_BOAT,
    "Black Bomb": UnitType.BLACK_BOMB,
    "Piperunner": UnitType.PIPERUNNER,
    "Gunboat": UnitType.GUNBOAT,
    "Oozium": UnitType.OOZIUM,
}


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


def _repair_cost(ut: UnitType, hp_internal: int) -> int:
    """Repair restores 2 display HP (=20 internal), capped at max=100.
    Cost = build_cost / 10 * (delta display HP)."""
    stats = UNIT_STATS[ut]
    delta_internal = min(20, 100 - hp_internal)
    if delta_internal <= 0:
        return 0
    delta_display = delta_internal // 10
    return stats.cost // 10 * delta_display


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, default=1635679)
    ap.add_argument("--from-env", type=int, default=0)
    ap.add_argument("--to-env", type=int, default=32)
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
    print()

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

        if env_i < args.from_env or env_i > args.to_env:
            continue

        # At end of env_i, the active player has rolled to the next.
        # Engine state reflects start-of-next-turn (post-income, post-repair).
        # Snapshot eligible-for-repair set: engine's current active-player units
        # that sit on owned bases of matching country.
        active = state.active_player
        active_awbw = engine_to_awbw[active]
        eng_eligible = []
        for u in state.units[active]:
            prop = state.get_property_at(*u.pos)
            if prop is None:
                continue
            if prop.owner != active:
                continue
            # Property must repair this unit type. The engine's resupply path
            # determines this — we approximate: airport for air, port for sea,
            # base/city/etc for ground.
            eng_eligible.append((u.unit_type.name, u.pos, u.hp))

        # PHP-side eligible: frames[env_i+1] is post-end-turn snapshot.
        if env_i + 1 >= len(frames):
            continue
        fa = frames[env_i + 1]
        php_active_country = pid_to_country.get(active_awbw)
        bld_by_pos = {(int(b['y']), int(b['x'])): b for b in (fa.get('buildings') or {}).values()}
        php_units_at_props = []
        for u in (fa.get('units') or {}).values():
            try:
                pl_id = int(u['players_id'])
            except (TypeError, ValueError, KeyError):
                continue
            if pl_id != active_awbw:
                continue
            pos = (int(u['y']), int(u['x']))
            b = bld_by_pos.get(pos)
            if b is None:
                continue
            info = get_terrain(int(b['terrain_id']))
            if not info or info.country_id != php_active_country:
                continue
            name = u.get("name") or "?"
            hp = u.get("hit_points")
            php_units_at_props.append((name, pos, hp))

        # Frames also encode the *pre*-repair HP at frame[env_i+1] (since the
        # AWBW snapshot is *after* the end_turn that triggered repair). So PHP
        # unit HPs we see here are POST-repair. To get the implied repair
        # cost, compare against frame[env_i] (pre-end) for the same units.
        fb = frames[env_i]
        php_units_pre = {}
        bld_pre = {(int(b['y']), int(b['x'])): b for b in (fb.get('buildings') or {}).values()}
        for u in (fb.get('units') or {}).values():
            try:
                pl_id = int(u['players_id'])
            except (TypeError, ValueError, KeyError):
                continue
            if pl_id != active_awbw:
                continue
            pos = (int(u['y']), int(u['x']))
            php_units_pre[pos] = (u.get("name"), u.get("hit_points"))

        php_implied_cost = 0
        for name, pos, post_hp in php_units_at_props:
            pre = php_units_pre.get(pos)
            if pre is None:
                continue
            try:
                pre_hp = float(pre[1])
                post_hpf = float(post_hp)
            except (TypeError, ValueError):
                continue
            ut = _PHP_NAME_TO_UT.get(name)
            if ut is None:
                continue
            # Display HP delta — AWBW awards 2 HP per repair, capped at 10.
            delta_display = max(0.0, min(post_hpf, 10.0) - pre_hp)
            if delta_display <= 0:
                continue
            stats = UNIT_STATS[ut]
            php_implied_cost += int(stats.cost // 10 * round(delta_display))

        # Engine implied cost: re-run repair logic against engine state pre-repair.
        # We approximate by computing what _grant_property_resupply would charge.
        eng_implied_cost = 0
        for name, _, hp in eng_eligible:
            try:
                ut = UnitType[name]
            except KeyError:
                continue
            eng_implied_cost += _repair_cost(ut, hp)

        if env_i < args.from_env:
            continue
        diff = eng_implied_cost - php_implied_cost
        marker = "  " if diff == 0 else "**"
        print(f"{marker} env={env_i:>3} day={day:>3} active=P{active} ({active_awbw})  "
              f"engine_repair_cost={eng_implied_cost:>5}  php_implied={php_implied_cost:>5}  "
              f"delta={diff:+d}")
        if diff != 0:
            print(f"     ENG eligible ({len(eng_eligible)}):")
            for name, pos, hp in sorted(eng_eligible):
                print(f"       {name:>10} {pos} hp={hp}")
            print(f"     PHP at-props ({len(php_units_at_props)}):")
            for name, pos, hp in sorted(php_units_at_props):
                print(f"       {name:>10} {pos} hp={hp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
