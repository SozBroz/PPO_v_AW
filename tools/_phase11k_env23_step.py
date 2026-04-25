#!/usr/bin/env python3
"""Step through env 23 (Lash day 12) and dump engine state at (7,9) after every action."""
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
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)

CATS = [
    ROOT / "data" / "amarriner_gl_std_catalog.json",
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
    ROOT / "data" / "amarriner_gl_colin_batch.json",
]
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _state_at(state, pos):
    cap = None
    for p in state.properties:
        if (p.row, p.col) == pos:
            cap = (p.owner, p.terrain_id, p.capture_points)
            break
    unit = None
    for s in (0, 1):
        for u in state.units[s]:
            if u.pos == pos:
                unit = (s, u.unit_type.name, u.hp, getattr(u, "unit_id", None), getattr(u, "is_alive", True))
                break
        if unit:
            break
    return cap, unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, default=1635679)
    ap.add_argument("--env", type=int, default=23)
    ap.add_argument("--row", type=int, default=7)
    ap.add_argument("--col", type=int, default=9)
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
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    target = (args.row, args.col)
    for env_i, (pid, day, actions) in enumerate(envs):
        if env_i + 1 < len(frames):
            pin = {}
            for u in (frames[env_i + 1].get("units") or {}).values():
                try:
                    uid = int(u["id"]); hp = float(u["hit_points"])
                except (TypeError, ValueError, KeyError):
                    continue
                pin[uid] = max(0, min(100, int(round(hp * 10))))
            state._oracle_post_envelope_units_by_id = pin
            def_hits = {}
            for o in actions:
                if isinstance(o, dict) and o.get("action") in ("Fire", "AttackSeam"):
                    ci = o.get("combatInfo")
                    if isinstance(ci, dict):
                        d = ci.get("defender")
                        if isinstance(d, dict):
                            try:
                                u = int(d.get("units_id"))
                                def_hits[u] = def_hits.get(u, 0) + 1
                            except (TypeError, ValueError):
                                pass
            state._oracle_post_envelope_multi_hit_defenders = {u for u, c in def_hits.items() if c > 1}

        if env_i == args.env:
            cap, unit = _state_at(state, target)
            print(f"PRE env {env_i}: prop={cap} unit_at_(7,9)={unit}")
            cap_710, u_710 = _state_at(state, (7, 10))
            print(f"             unit_at_(7,10)={u_710}")
            for ai, obj in enumerate(actions):
                kind = obj.get("action") if isinstance(obj, dict) else "?"
                if ai == 6:
                    cap_710, u_710 = _state_at(state, (7, 10))
                    cap_69, u_69 = _state_at(state, (6, 9))
                    cap_68, u_68 = _state_at(state, (6, 8))
                    cap_78, u_78 = _state_at(state, (7, 8))
                    print(f"   pre ai=6: (7,10)={u_710}  (6,9)={u_69}  (6,8)={u_68}  (7,8)={u_78}")
                apply_oracle_action_json(state, obj, awbw_to_engine,
                                         envelope_awbw_player_id=pid)
                cap, unit = _state_at(state, target)
                print(f"  ai={ai} {kind:>10}: prop_at_(7,9)={cap} unit_at_(7,9)={unit}")
                if ai == 6:
                    cap_710, u_710 = _state_at(state, (7, 10))
                    cap_69, u_69 = _state_at(state, (6, 9))
                    cap_68, u_68 = _state_at(state, (6, 8))
                    cap_78, u_78 = _state_at(state, (7, 8))
                    print(f"   post ai=6: (7,10)={u_710}  (6,9)={u_69}  (6,8)={u_68}  (7,8)={u_78}")
            return 0

        for obj in actions:
            apply_oracle_action_json(state, obj, awbw_to_engine,
                                     envelope_awbw_player_id=pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
