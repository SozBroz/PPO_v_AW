#!/usr/bin/env python3
"""Check property ownership and unit HP at (17,15) and (13,13) in PHP
frames around env 24-25 of gid 1635679 to test the engine-vs-PHP repair
divergence hypothesis."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.diff_replay_zips import load_replay
from engine.terrain import get_terrain

zpath = ROOT / "replays" / "amarriner_gl" / "1635679.zip"
frames = load_replay(zpath)
TARGETS = [(17, 15), (13, 13), (7, 9), (1, 18), (4, 20)]

for env_i in (23, 24, 25, 26, 27):
    snap = env_i
    if snap >= len(frames):
        continue
    f = frames[snap]
    print(f"\n=== frame[{snap}] day={f.get('day')} turn={f.get('turn')} ===")
    bld = {(int(b['y']), int(b['x'])): b for b in (f.get('buildings') or {}).values()}
    units = {(int(u['y']), int(u['x'])): u for u in (f.get('units') or {}).values()}
    for (y, x) in TARGETS:
        b = bld.get((y, x))
        u = units.get((y, x))
        b_str = (f"terrain_id={b.get('terrain_id')} owner_pid={b.get('players_id')} "
                 f"team={b.get('team')} kind={get_terrain(int(b['terrain_id'])).name}"
                 if b else "no building")
        u_str = (f"name={u.get('name')} pid={u.get('players_id')} hp={u.get('hit_points')} "
                 f"id={u.get('id')}" if u else "no unit")
        print(f"  ({y},{x}) BUILD: {b_str}")
        print(f"        UNIT:  {u_str}")
