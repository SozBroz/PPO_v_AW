"""Debug compare_units for Game 1553655 - check all snapshots around the divergence."""
import sys
sys.path.insert(0, 'D:/awbw')

from tools.oracle_zip_replay import load_replay, map_snapshot_player_ids_to_engine
from tools.replay_snapshot_compare import compare_snapshot_to_engine
from engine.game import GameState

GAME_ID = 1553655
ZIP_PATH = f'D:/awbw/replays/amarriner_gl/{GAME_ID}.zip'

# Load replay
php_snapshots = load_replay(ZIP_PATH)
print(f"PHP snapshots: {len(php_snapshots)}")

# Get mapping from snapshot 0
snap0 = php_snapshots[0]
co_p0, co_p1 = 9, 2  # from catalog
awbw_to_engine = map_snapshot_player_ids_to_engine(snap0, co_p0, co_p1)
print(f"Mapping: {awbw_to_engine}")

# Check snapshots 10-15 for (1, 6, 7)
print(f"\nChecking snapshots 10-15 for (1, 6, 7):")
for idx in range(10, min(16, len(php_snapshots))):
    php = php_snapshots[idx]
    units = php.get('units', {})
    found = False
    for k, u in (units if isinstance(units, dict) else {}).items():
        if not isinstance(u, dict): continue
        col, row = int(u['x']), int(u['y'])
        if row == 6 and col == 7:
            pid = int(u['players_id'])
            eng_seat = awbw_to_engine.get(pid, '?')
            carried = str(u.get('carried', 'N')).upper()
            name = u.get('name', '?')
            print(f"  Snapshot {idx}: Found {name} at P{eng_seat} (6,7) carried={carried}")
            found = True
    if not found:
        print(f"  Snapshot {idx}: (1, 6, 7) NOT found")
