#!/usr/bin/env python3
"""Phase 11J-FINAL — dump env 25 of gid 1628849 action-by-action with funds.

Wraps apply_oracle_action_json so we record before/after engine funds and any
errors. Intended to identify the EXACT 200g divergence preceding the failing
B_COPTER build at (10,18).
"""
from __future__ import annotations

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

GID = 1628849
TARGET_ENV = 25  # Koal day 13

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
EXTRAS = ROOT / "data" / "amarriner_gl_extras_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def main():
    by_id = {}
    for cat in (CATALOG, EXTRAS):
        if cat.exists():
            d = json.loads(cat.read_text(encoding="utf-8"))
            for g in (d.get("games") or {}).values():
                if isinstance(g, dict) and "games_id" in g:
                    by_id[int(g["games_id"])] = g
    meta = by_id[GID]
    random.seed(_seed_for_game(CANONICAL_SEED, GID))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
    zpath = ZIPS / f"{GID}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    print(f"awbw_to_engine={awbw_to_engine}")

    for env_i, (pid, day, actions) in enumerate(envs):
        if env_i < TARGET_ENV:
            try:
                for obj in actions:
                    apply_oracle_action_json(state, obj, awbw_to_engine,
                                             envelope_awbw_player_id=pid)
            except UnsupportedOracleAction as e:
                print(f"  early oracle_gap@{env_i}: {e}")
                return
        elif env_i == TARGET_ENV:
            print(f"\n=== ENV {env_i} pid={pid} day={day} actions={len(actions)} ===")
            print(f"  start funds: P0={int(state.funds[0])} P1={int(state.funds[1])}")
            for j, obj in enumerate(actions):
                kind = obj.get("action") or obj.get("type")
                f0_before = int(state.funds[0])
                f1_before = int(state.funds[1])
                try:
                    apply_oracle_action_json(state, obj, awbw_to_engine,
                                             envelope_awbw_player_id=pid)
                    f0_after = int(state.funds[0])
                    f1_after = int(state.funds[1])
                    df0 = f0_after - f0_before
                    df1 = f1_after - f1_before
                    detail = ""
                    if kind == "Build":
                        unit = obj.get("unit") or obj.get("Build", {}).get("unit") or obj.get("data", {}).get("newUnit", {})
                        detail = f" Build={unit}"
                    elif kind == "Capt":
                        detail = f" Capt={obj.get('Capt') or obj.get('data') or obj}"
                    elif kind == "Move":
                        detail = f" Move={obj.get('paths') or obj.get('Move') or obj.get('data')}"
                    elif kind == "Fire":
                        detail = f" Fire={obj.get('Fire') or obj.get('data')}"
                    elif kind == "Power":
                        detail = f" Power={obj.get('Power') or obj.get('data') or obj}"
                    print(f"  [{j:2}] {kind:10} P0={f0_after:>6} P1={f1_after:>6} dP0={df0:+5} dP1={df1:+5}{detail}")
                except UnsupportedOracleAction as e:
                    print(f"  [{j:2}] {kind:10} *** REFUSED: {e}")
                    print(f"       state at refusal: P0={int(state.funds[0])} P1={int(state.funds[1])}")
                    print(f"       full action obj: {json.dumps(obj, default=str)[:600]}")
                    return
                except Exception as e:
                    print(f"  [{j:2}] {kind:10} EXC {type(e).__name__}: {e}")
                    return
            return


if __name__ == "__main__":
    main()
