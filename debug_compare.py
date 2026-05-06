"""Debug compare_units for Game 1553655."""
import sys
sys.path.insert(0, 'D:/awbw')

from tools.oracle_zip_replay import load_replay, map_snapshot_player_ids_to_engine
from tools.replay_snapshot_compare import compare_units, compare_snapshot_to_engine
from engine.game import GameState

GAME_ID = 1553655
ZIP_PATH = f'D:/awbw/replays/amarriner_gl/{GAME_ID}.zip'

# Load replay
php_snapshots = load_replay(ZIP_PATH)
snap0 = php_snapshots[0]

# Get mapping
co_p0, co_p1 = 9, 2  # from catalog
awbw_to_engine = map_snapshot_player_ids_to_engine(snap0, co_p0, co_p1)
print(f"Mapping: {awbw_to_engine}")

# Check snapshot 12 (after envelope 11, approx_day=9)
target_idx = min(12, len(php_snapshots) - 1)
php = php_snapshots[target_idx]

# Run compare_units manually
print(f"\nComparing snapshot {target_idx}:")
units = php.get('units', {})
print(f"PHP units count: {len(units) if isinstance(units, dict) else 0}")

# Build php_by_tile manually
php_by_tile = {}
for k, u in (units if isinstance(units, dict) else {}).items():
    if not isinstance(u, dict): continue
    carried = str(u.get('carried', 'N')).upper()
    if carried in ('Y', 'YES', '1', 'TRUE'):
        continue
    col, row = int(u['x']), int(u['y'])
    pid = int(u['players_id'])
    eng_seat = awbw_to_engine.get(pid, '?')
    if eng_seat == '?':
        print(f"  WARNING: players_id={pid} not in mapping!")
        continue
    key = (eng_seat, row, col)
    if key in php_by_tile:
        print(f"  DUPLICATE at {key}")
    php_by_tile[key] = u
    print(f"  P{eng_seat} at ({row},{col}) {u.get('name', '?')} carried={carried}")

print(f"\nphp_by_tile keys: {sorted(php_by_tile.keys())}")
print(f"Is (1, 6, 7) in php_by_tile? {(1, 6, 7) in php_by_tile}")
