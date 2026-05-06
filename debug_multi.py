"""Debug multi desync entries - find why PHP has units engine doesn't."""
import sys
sys.path.insert(0, 'D:/awbw')

from tools.oracle_zip_replay import load_replay

GAME_ID = 1553655
ZIP_PATH = f'D:/awbw/replays/amarriner_gl/{GAME_ID}.zip'

print(f"=== Debugging Game {GAME_ID} ===")

# Load replay - returns list of PHP snapshots
php_snapshots = load_replay(ZIP_PATH)
print(f"PHP snapshots: {len(php_snapshots)}")

# Check snapshot 11 (from the log: approx_day=9, approx_envelope_index=12)
# Snapshots are 0-indexed.  Envelope i -> compare to snapshot i+1
# So if approx_envelope_index=12, we compare to snapshot 12 or 13?
# Let's check snapshot 12 (after envelope 11)
target_idx = min(12, len(php_snapshots) - 1)
php = php_snapshots[target_idx]
print(f"\nSnapshot {target_idx}:")

# Get the awbw_to_engine mapping (simulate what audit does)
# From the log: co_p0_id=9, co_p1_id=2
# First mover is P0 (from the game)
awbw_to_engine = {}
players = php.get('players', {})
for k, p in players.items():
    if not isinstance(p, dict): continue
    pid = int(p.get('id', 0))
    order = int(p.get('order', 0))
    cid = int(p.get('co_id', 0))
    # co_p0=9 is order=0, co_p1=2 is order=1
    seat = 0 if order == 0 else 1
    awbw_to_engine[pid] = seat

print(f"awbw_to_engine mapping: {awbw_to_engine}")

# Now check all units
units = php.get('units', {})
print(f"\nAll units at (6,7) or (7,6):")
for k, u in (units if isinstance(units, dict) else {}).items():
    if not isinstance(u, dict): continue
    col, row = int(u['x']), int(u['y'])
    if (row == 6 and col == 7) or (row == 7 and col == 6):
        pid = int(u.get('players_id', 0))
        eng_seat = awbw_to_engine.get(pid, '?')
        carried = str(u.get('carried', 'N')).upper()
        name = u.get('name', '?')
        print(f"  {name} at ({row},{col}) pid={pid} -> seat={eng_seat} carried={carried}")

# Also list ALL unique (seat, row, col) positions from PHP
print(f"\nAll PHP (seat, row, col) positions:")
php_positions = set()
for k, u in (units if isinstance(units, dict) else {}).items():
    if not isinstance(u, dict): continue
    pid = int(u.get('players_id', 0))
    col, row = int(u['x']), int(u['y'])
    eng_seat = awbw_to_engine.get(pid, '?')
    carried = str(u.get('carried', 'N')).upper()
    if carried not in ('Y', 'YES', '1', 'TRUE'):
        php_positions.add((eng_seat, row, col))

for s, r, c in sorted(php_positions):
    print(f"  P{s} ({r},{c})")
