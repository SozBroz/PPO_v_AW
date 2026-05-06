#!/usr/bin/env python3
"""Directly test what happens in _audit_one for game 1553655"""
import sys
sys.path.insert(0, 'D:/awbw')

from pathlib import Path
import json
from tools.desync_audit import _audit_one

# Load catalog
with open('D:/awbw/data/amarriner_gl_std_catalog.json') as f:
    catalog_data = json.load(f)

# 'games' is a dict keyed by games_id as strings
games_dict = catalog_data['games']
print(f"Type of games: {type(games_dict)}")
print(f"Keys (first 5): {list(games_dict.keys())[:5]}")

# Get meta for game 1553655
meta = games_dict.get('1553655')
if meta is None:
    print("Game 1553655 not found in catalog")
    sys.exit(1)

print(f"Found meta for 1553655: games_id={meta.get('games_id')}")

try:
    row = _audit_one(
        games_id=1553655,
        zip_path=Path('D:/awbw/replays/amarriner_gl/1553655.zip'),
        meta=meta,
        map_pool=Path('D:/awbw/data/gl_map_pool.json'),
        maps_dir=Path('D:/awbw/data/maps'),
        seed=42,
        enable_state_mismatch=True,
        state_mismatch_hp_tolerance=0,
    )
    print(f"Result: cls={row.cls}, status={row.status}")
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
