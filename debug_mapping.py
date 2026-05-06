"""Debug mapping for Game 1553655."""
import sys
sys.path.insert(0, 'D:/awbw')

from tools.oracle_zip_replay import load_replay, map_snapshot_player_ids_to_engine

GAME_ID = 1553655
ZIP_PATH = f'D:/awbw/replays/amarriner_gl/{GAME_ID}.zip'

# Load replay
php_snapshots = load_replay(ZIP_PATH)
snap0 = php_snapshots[0]

# Simulate what audit does: co_p0_id=9, co_p1_id=2
co_p0 = 9
co_p1 = 2

# Get mapping
try:
    mapping = map_snapshot_player_ids_to_engine(snap0, co_p0, co_p1)
    print(f"Mapping: {mapping}")
    
    # Check players in snap0
    players = snap0.get('players', {})
    for k, p in players.items():
        if not isinstance(p, dict): continue
        pid = int(p.get('id', 0))
        order = int(p.get('order', 0))
        cid = int(p.get('co_id', 0))
        print(f"  Player {k}: id={pid}, order={order}, co_id={cid}")
        
except Exception as e:
    print(f"Error: {e}")
