#!/usr/bin/env python3
"""Per-action funds drill for a single envelope of a single gid.

Replays envelopes 0..target_env-1 fully, then steps through target_env
action-by-action and prints engine funds for both seats after each action.
The exact action that introduces drift is the bug suspect.
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

from engine.game import make_initial_state
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
EXTRA = (
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
    ROOT / "data" / "amarriner_gl_colin_batch.json",
)
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env", type=int, required=True, help="Envelope index to drill")
    ap.add_argument("--show-all-envs", action="store_true",
                    help="Also print funds before each prior envelope")
    args = ap.parse_args()

    by_id: dict[int, dict] = {}
    for cp in (CATALOG, *EXTRA):
        if not cp.exists():
            continue
        cat = json.loads(cp.read_text(encoding="utf-8"))
        for g in (cat.get("games") or {}).values():
            if isinstance(g, dict) and "games_id" in g:
                by_id.setdefault(int(g["games_id"]), g)
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

    print(f"gid={args.gid} co0={co0} co1={co1} target_env={args.env}")
    print()

    # Replay envelopes [0, args.env)
    for env_i in range(args.env):
        pid, day, actions = envs[env_i]
        for obj in actions:
            apply_oracle_action_json(state, obj, awbw_to_engine,
                                     envelope_awbw_player_id=pid)
        if args.show_all_envs:
            print(f"  after env[{env_i:>2}] pid={pid} day={day} "
                  f"funds=[{state.funds[0]},{state.funds[1]}]")

    # Drill target envelope
    pid, day, actions = envs[args.env]
    print(f"=== TARGET env[{args.env}] pid={pid} day={day} active_eng={state.active_player} ===")
    print(f"  initial funds: P0={state.funds[0]} P1={state.funds[1]}")
    # Dump P1 infantries with their HP and pos at envelope start
    print("  P1 INF/MECH at env start:")
    for u in list(state.units[1]):
        if not getattr(u, "is_alive", True):
            continue
        if u.unit_type.name in ("INFANTRY", "MECH"):
            print(f"    type={u.unit_type.name} pos={u.pos} hp={u.hp} (display {u.display_hp}) unit_id={u.unit_id}")
    print()
    for ai, obj in enumerate(actions):
        kind = obj.get("action") or "?"
        before0, before1 = state.funds[0], state.funds[1]
        # Brief obj dump
        info_parts = []
        for k in ("Move", "Build", "Capt", "Fire", "Power", "Repair", "End",
                  "type", "x", "y", "fromX", "fromY", "tile", "cost", "unit",
                  "buildingID", "buildingType", "name", "playerID"):
            if k in obj:
                v = obj[k]
                if isinstance(v, dict):
                    s = json.dumps({k2: v2 for k2, v2 in list(v.items())[:6]})
                    info_parts.append(f"{k}={s}")
                else:
                    info_parts.append(f"{k}={v}")
        try:
            apply_oracle_action_json(state, obj, awbw_to_engine,
                                     envelope_awbw_player_id=pid)
        except UnsupportedOracleAction as e:
            print(f"  [{ai:>2}] {kind:<14} ORACLE_GAP: {e}")
            print(f"       state pre: P0={before0} P1={before1}")
            return 0
        d0 = state.funds[0] - before0
        d1 = state.funds[1] - before1
        marker = "*" if (d0 != 0 or d1 != 0) else " "
        info = " ".join(info_parts)[:120]
        print(f"  [{ai:>2}]{marker}{kind:<14} d=[{d0:+5},{d1:+5}] -> [{state.funds[0]:>6},{state.funds[1]:>6}]  {info}")
        if kind == "Join":
            recent = [e for e in state.game_log[-15:] if isinstance(e, dict)]
            print(f"      RECENT LOG ENTRIES (last 15):")
            for e in recent:
                t = e.get("type")
                print(f"        type={t} keys={list(e.keys())[:8]} entry={json.dumps(e)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
