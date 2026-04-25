"""Inspect PHP frame 40 (start of Drake env 40, day 21) for property ownership
and unit data at (11,16) and other suspect tiles.

Goal: confirm whether PHP property at (11,16) is owned by Rachel (P1) and what
PHP's pre-tick view of the unit there is. Compare to engine's view.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    parse_p_envelopes_from_zip,
    map_snapshot_player_ids_to_engine,
)


def main() -> int:
    gid = 1607045
    co0, co1 = 5, 28
    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"

    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)

    print(f"awbw_to_engine: {awbw_to_engine}")
    p1_awbw_ids = [aid for aid, eng in awbw_to_engine.items() if eng == 1]
    print(f"P1 (Rachel) awbw player ids: {p1_awbw_ids}")

    # Frame 40 = state at start of env 40 (= post env 39).
    for fi in (39, 40, 41):
        if fi >= len(frames):
            continue
        frame = frames[fi]
        # Top-level keys
        print(f"\n=== Frame {fi} top-level keys: {sorted(frame.keys())[:30]} ===")
        # Try buildings
        buildings = frame.get("buildings") or frame.get("properties") or {}
        if isinstance(buildings, dict):
            bs = list(buildings.values())
        elif isinstance(buildings, list):
            bs = buildings
        else:
            bs = []
        # Find props at watched tiles
        watched = [(11, 16), (10, 12), (12, 18), (8, 7), (8, 9), (9, 8)]
        for (x, y) in watched:
            hits = [b for b in bs if isinstance(b, dict) and (
                int(b.get("x", -1)) == x and int(b.get("y", -1)) == y
            )]
            for b in hits:
                ownr = b.get("buildings_team") or b.get("team") or b.get("players_id") or b.get("owner")
                kind = b.get("terrain_name") or b.get("name") or b.get("buildings_terrain_id") or b.get("type")
                print(f"  prop @ ({x},{y}): owner_raw={ownr!r} kind={kind!r} keys={list(b.keys())[:10]}")
        # Find units at watched tiles for P1
        units = frame.get("units") or {}
        if isinstance(units, dict):
            us = list(units.values())
        else:
            us = units
        for (x, y) in watched:
            hits = [u for u in us if isinstance(u, dict) and (
                int(u.get("x", -1)) == x and int(u.get("y", -1)) == y
            )]
            for u in hits:
                pid = u.get("players_id")
                eng = awbw_to_engine.get(int(pid)) if pid is not None else None
                print(f"  unit @ ({x},{y}): id={u.get('id')} hp={u.get('hit_points')} name={u.get('name')} pid={pid} eng_seat={eng}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
