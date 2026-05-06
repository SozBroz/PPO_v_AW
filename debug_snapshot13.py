"""Debug - check snapshot 13 for (1, 6, 7)."""
import sys
sys.path.insert(0, 'D:/awbw')

from tools.oracle_zip_replay import load_replay, map_snapshot_player_ids_to_engine

GAME_ID = 1553655
ZIP_PATH = f'D:/awbw/replays/amarriner_gl/{GAME_ID}.zip'

# Load replay
php_snapshots = load_replay(ZIP_PATH)
print(f"PHP snapshots: {len(php_snapshots)}")

# Get mapping
snap0 = php_snapshots[0]
awbw_to_engine = map_snapshot_player_ids_to_engine(snap0, 9, 2)
print(f"Mapping: {awbw_to_engine}")

# Check snapshot 13 (after envelope 12)
target_idx = 13
if target_idx < len(php_snapshots):
    php = php_snapshots[target_idx]
    units = php.get('units', {})
    print(f"\nSnapshot {target_idx} - units count: {len(units) if isinstance(units, dict) else 0}")
    
    # Build php_by_tile like compare_units does
    php_by_tile = {}
    for k, u in (units if isinstance(units, dict) else {}).items():
        if not isinstance(u, dict): continue
        carried = str(u.get('carried', 'N')).upper()
        if carried in ('Y', 'YES', '1', 'TRUE'):
            continue
        col, row = int(u['x']), int(u['y'])
        pid = int(u['players_id'])
        eng_seat = awbw_to_engine.get(pid)
        if eng_seat is None:
            print(f"  WARNING: players_id={pid} not in mapping!")
            continue
        key = (eng_seat, row, col)
        php_by_tile[key] = u
    
    print(f"\nphp_by_tile keys containing (6,7):")
    for k in php_by_tile:
        if k[1] == 6 and k[2] == 7:
            print(f"  FOUND: {k} -> {php_by_tile[k].get('name', '?')}")
    
    print(f"\nAll P1 units in php_by_tile:")
    for k, u in sorted(php_by_tile.items()):
        if k[0] == 1:
            print(f"  P1 ({k[1]},{k[2]}) {u.get('name', '?')} carried={str(u.get('carried', 'N')).upper()}")
