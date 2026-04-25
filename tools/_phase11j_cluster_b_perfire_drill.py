#!/usr/bin/env python3
"""Phase 11J-CLUSTER-B-SHIP — per-attack engine vs PHP defender HP delta drill.

For a target gid and envelope, run all envelopes up to and including
the target. Inside the target envelope, intercept each ``Fire`` /
``AttackSeam`` action: capture defender (and attacker) engine HP
immediately AFTER the action lands, then compare to the PHP frame at
``frame[env_i + 1]`` (post-envelope snapshot).

Goal: surface attacks where engine-side defender HP < PHP-side defender HP.
Those are the strikes where ``_oracle_combat_damage_override`` should pin
but isn't.
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


def _php_units_at(frame, awbw_pid: int):
    """Return dict ``{(y, x) -> hp_internal}`` for PHP units of awbw_pid."""
    out = {}
    for u in (frame.get("units") or {}).values():
        try:
            if int(u["players_id"]) != awbw_pid:
                continue
            pos = (int(u["y"]), int(u["x"]))
            disp = float(u["hit_points"])
        except (TypeError, ValueError, KeyError):
            continue
        out[pos] = round(disp * 10)
    return out


def _fire_signature(obj):
    """Pull (defender_pos, attacker_pos, defender_id, attacker_id) from a Fire dict."""
    fire = obj.get("Fire") if isinstance(obj, dict) else None
    if not isinstance(fire, dict):
        return None
    civ = fire.get("combatInfoVision") or {}
    if not isinstance(civ, dict) or not civ:
        return None
    # take the first vision bucket (any)
    bucket = next(iter(civ.values()))
    if not isinstance(bucket, dict):
        return None
    ci = bucket.get("combatInfo") or {}
    att = ci.get("attacker") or {}
    deft = ci.get("defender") or {}
    if not isinstance(att, dict) or not isinstance(deft, dict):
        return None
    try:
        d_pos = (int(deft["units_y"]), int(deft["units_x"]))
        a_pos = (int(att["units_y"]), int(att["units_x"]))
    except (TypeError, ValueError, KeyError):
        return None
    try:
        d_hp = int(deft.get("units_hit_points")) if deft.get("units_hit_points") is not None else None
    except (TypeError, ValueError):
        d_hp = None
    try:
        a_hp = int(att.get("units_hit_points")) if att.get("units_hit_points") is not None else None
    except (TypeError, ValueError):
        a_hp = None
    try:
        d_id = int(deft.get("units_id")) if deft.get("units_id") is not None else None
    except (TypeError, ValueError):
        d_id = None
    try:
        a_id = int(att.get("units_id")) if att.get("units_id") is not None else None
    except (TypeError, ValueError):
        a_id = None
    return {"d_pos": d_pos, "a_pos": a_pos, "d_hp_php": d_hp, "a_hp_php": a_hp,
            "d_id": d_id, "a_id": a_id}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env", type=int, required=True,
                    help="Envelope to drill (per-Fire defender HP)")
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

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    # PHP defender pid for the target envelope = the *opponent* of the
    # acting player in that envelope. We report engine vs PHP for the
    # defender only (the attacker is the acting player).
    target_env = args.env
    target_pid, target_day, target_actions = envs[target_env]
    actor_eng = awbw_to_engine[target_pid]
    opp_eng = 1 - actor_eng
    opp_pid = engine_to_awbw[opp_eng]
    php_after_units = _php_units_at(frames[target_env + 1], opp_pid)
    php_before_units = _php_units_at(frames[target_env], opp_pid)

    print(f"gid={args.gid} env={target_env} actor_eng={actor_eng} opp_eng={opp_eng}")
    print(f"actor_awbw={target_pid} opp_awbw={opp_pid} co0={co0} co1={co1}")
    print(f"actions={ {a.get('action'): None for a in target_actions if isinstance(a, dict)} }")
    print()

    # Replay envelopes [0, target_env)
    for env_i in range(target_env):
        for obj in envs[env_i][2]:
            try:
                apply_oracle_action_json(state, obj, awbw_to_engine,
                                         envelope_awbw_player_id=envs[env_i][0])
            except UnsupportedOracleAction as e:
                print(f"oracle_gap@env={env_i}: {e}")
                return 1

    # Now step actions in the target envelope one-by-one, intercepting Fire.
    print(f"{'idx':>3} {'kind':>10} {'a_pos':>10} {'d_pos':>10} "
          f"{'d_hp_pre':>9} {'d_hp_post_eng':>14} {'d_hp_php_after':>15} "
          f"{'delta_eng_php':>14} note")
    for j, obj in enumerate(target_actions):
        kind = obj.get("action") if isinstance(obj, dict) else None
        sig = None
        defender_pre_hp = None
        if kind == "Fire":
            sig = _fire_signature(obj)
            if sig is not None:
                d_pos = sig["d_pos"]
                u = state.get_unit_at(*d_pos)
                if u is not None and u.is_alive:
                    defender_pre_hp = int(u.hp)
        try:
            apply_oracle_action_json(state, obj, awbw_to_engine,
                                     envelope_awbw_player_id=target_pid)
        except UnsupportedOracleAction as e:
            print(f"{j:>3} {kind:>10} oracle_gap: {e}")
            return 1
        if kind == "Fire" and sig is not None:
            d_pos = sig["d_pos"]
            u = state.get_unit_at(*d_pos)
            d_hp_post_eng = int(u.hp) if (u is not None and u.is_alive) else 0
            d_hp_php_after = php_after_units.get(d_pos, "n/a")
            delta = "n/a"
            if isinstance(d_hp_php_after, int):
                delta = d_hp_post_eng - d_hp_php_after
            note = ""
            if isinstance(d_hp_php_after, int) and d_hp_post_eng != d_hp_php_after:
                note = f"DRIFT (PHP_pre={php_before_units.get(d_pos, 'n/a')})"
            print(f"{j:>3} {kind:>10} {str(sig['a_pos']):>10} {str(d_pos):>10} "
                  f"{str(defender_pre_hp):>9} {d_hp_post_eng:>14} "
                  f"{str(d_hp_php_after):>15} {str(delta):>14} {note}")
        elif kind in ("Fire",):
            print(f"{j:>3} {kind:>10} (no combatInfo signature)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
