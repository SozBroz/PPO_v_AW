#!/usr/bin/env python3
"""Phase 11K-PROP-HISTORY — trace ownership of a single property
position across all PHP frames AND all engine envelope ends.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, default=1635679)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--col", type=int, required=True)
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
    engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    target_row, target_col = args.row, args.col
    print(f"gid={args.gid} target=({target_row},{target_col})  awbw_to_engine={awbw_to_engine}")
    print()

    # Look through all PHP frames at the building at (target_row, target_col).
    print("PHP frame ownership of property (by frame index):")
    php_owner_per_frame = []
    for fi, f in enumerate(frames):
        b = None
        for bb in (f.get('buildings') or {}).values():
            try:
                bx, by = int(bb['x']), int(bb['y'])
            except (TypeError, ValueError, KeyError):
                continue
            if (by, bx) == (target_row, target_col):
                b = bb
                break
        if b is None:
            php_owner_per_frame.append(None)
            continue
        try:
            tid = int(b['terrain_id'])
        except (TypeError, ValueError, KeyError):
            tid = None
        info = get_terrain(tid) if tid is not None else None
        country_id = info.country_id if info else None
        php_owner_per_frame.append((country_id, tid))

    # Also note PHP unit on this tile per frame
    print("frame | php_country | engine_owner | unit_at_tile (player, name, hp, capture_pts)")
    print("-" * 100)

    # Apply envelopes one by one and check engine state.
    eng_engine_owner_per_env_end = []
    eng_unit_per_env_end = []

    for env_i, (pid, day, actions) in enumerate(envs):
        # Init
        if env_i == 0:
            # Print initial frame
            f = frames[0]
            unit_str = "-"
            for u in (f.get('units') or {}).values():
                try:
                    if (int(u['y']), int(u['x'])) == (target_row, target_col):
                        unit_str = f"pid={u.get('players_id')} {u.get('name')} hp={u.get('hit_points')}"
                        break
                except (TypeError, ValueError, KeyError):
                    pass
            # engine state pre-actions
            eng_owner = None
            for p in state.properties:
                if (p.row, p.col) == (target_row, target_col):
                    eng_owner = (p.owner, p.terrain_id, p.capture_points)
                    break
            print(f"f0     |  {php_owner_per_frame[0]}     |  {eng_owner}    |  {unit_str}")

        # Apply this envelope
        for obj in actions:
            apply_oracle_action_json(state, obj, awbw_to_engine,
                                     envelope_awbw_player_id=pid)
        # After env_i applied, compare to frame[env_i + 1]
        snap_i = env_i + 1
        if snap_i >= len(frames):
            break
        f = frames[snap_i]
        unit_str = "-"
        for u in (f.get('units') or {}).values():
            try:
                if (int(u['y']), int(u['x'])) == (target_row, target_col):
                    unit_str = f"pid={u.get('players_id')} {u.get('name')} hp={u.get('hit_points')} cap={u.get('capture')}"
                    break
            except (TypeError, ValueError, KeyError):
                pass
        eng_owner = None
        for p in state.properties:
            if (p.row, p.col) == (target_row, target_col):
                eng_owner = (p.owner, p.terrain_id, p.capture_points)
                break
        # engine unit at tile
        eng_unit_str = "-"
        for seat in (0, 1):
            for u in state.units[seat]:
                if u.pos == (target_row, target_col):
                    eng_unit_str = f"seat={seat} {u.unit_type.name} hp={u.hp} cap={getattr(u,'capture_progress',None)}"
                    break
            if eng_unit_str != "-":
                break
        print(f"f{snap_i:<5} | php_country={php_owner_per_frame[snap_i]} | eng={eng_owner} | PHP_unit: {unit_str}  | ENG_unit: {eng_unit_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
