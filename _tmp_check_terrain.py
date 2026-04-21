"""Check terrain at (8,2) for game 1631302."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.desync_audit import CATALOG_DEFAULT, MAP_POOL_DEFAULT, MAPS_DIR_DEFAULT, _load_catalog
from engine.map_loader import load_map

cat = _load_catalog(CATALOG_DEFAULT)
games = cat.get('games') or {}
by_id = {int(g['games_id']): g for g in games.values() if isinstance(g, dict)}
meta = by_id[1631302]
map_id = meta['map_id']
print('map_id:', map_id)

m = load_map(map_id, MAP_POOL_DEFAULT, MAPS_DIR_DEFAULT)
print('size: H', m.height, 'W', m.width)
# Show rows 0-3 and 7-12 cols 0-12 for context
for y in list(range(0, 4)) + list(range(7, 12)):
    row = []
    for x in range(0, 12):
        try:
            t = m.terrain[y][x]
            row.append(f'{t:3d}')
        except Exception:
            row.append('???')
    print(f'  y={y}: {row}')

print('\nProperties (all):')
for p in m.properties:
    if p.is_port:
        print(f'  PORT (r={p.row},c={p.col}) terrain_id={p.terrain_id} owner={p.owner}')
