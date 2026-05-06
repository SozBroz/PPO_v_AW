#!/usr/bin/env python3
"""Test _audit_one for game 1553655"""
import sys
sys.path.insert(0, 'D:/awbw')

from pathlib import Path
from tools.desync_audit import _audit_one

# Load the catalog to get meta for game 1553655
import json
catalog = json.load(open('D:/awbw/data/amarriner_gl_std_catalog.json'))
for game in catalog['games']:
    if game['games_id'] == 1553655:
        meta = game
        break

zip_path = Path('D:/awbw/replays/amarriner_gl/1553655.zip')
map_pool = Path('D:/awbw/data/gl_map_pool.json')
maps_dir = Path('D:/awbw/data/maps')

print(f'Testing game 1553655...')
try:
    row = _audit_one(
        games_id=1553655,
        zip_path=zip_path,
        meta=meta,
        map_pool=map_pool,
        maps_dir=maps_dir,
        seed=42,
        enable_state_mismatch=True,
        state_mismatch_hp_tolerance=0,
    )
    print(f'Result: {row.cls}')
except Exception as e:
    print(f'Exception: {e}')
    import traceback
    traceback.print_exc()
