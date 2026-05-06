"""Quick diagnostic: check pairing for game 1553655."""
import sys
sys.path.insert(0, 'D:/awbw')

from tools.oracle_zip_replay import load_replay, parse_p_envelopes_from_zip
from tools.replay_snapshot_compare import replay_snapshot_pairing

GAME_ID = 1553655
ZIP_PATH = 'D:/awbw/replays/amarriner_gl/{}.zip'.format(GAME_ID)

# Load replay (returns PHP snapshots)
php_snapshots = load_replay(ZIP_PATH)
print(f"PHP snapshots: {len(php_snapshots)}")

# Get envelopes
envelopes = parse_p_envelopes_from_zip(ZIP_PATH)
print(f"Envelopes: {len(envelopes)}")

# Check pairing
pairing = replay_snapshot_pairing(len(php_snapshots), len(envelopes))
print(f"Pairing mode: {pairing}")

# For trailing: N snapshots for N-1 envelopes
# After envelope i, compare to snapshot i+1
if pairing == "trailing":
    print(f"Trailing mode: {len(php_snapshots)} snapshots, {len(envelopes)} envelopes")
    print(f"After envelope i, compare to snapshot {i+1}")
    print(f"Last comparison: after envelope {len(envelopes)-1}, compare to snapshot {len(envelopes)}")
elif pairing == "tight":
    print(f"Tight mode: {len(php_snapshots)} snapshots, {len(envelopes)} envelopes")
    print(f"After envelope i (i < N-1), compare to snapshot i+1")
    print(f"Last envelope {len(envelopes)-1} has no post-frame to compare")
