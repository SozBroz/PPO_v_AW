"""Run engine for game 1553655 and check engine state."""
import sys
sys.path.insert(0, 'D:/awbw')

from tools.oracle_zip_replay import load_replay, parse_p_envelopes_from_zip, map_snapshot_player_ids_to_engine, apply_oracle_action_json
from engine.game import make_initial_state
from engine.map_loader import load_map
import json
from pathlib import Path

GAME_ID = 1553655
ZIP_PATH = Path('D:/awbw/replays/amarriner_gl/{}.zip'.format(GAME_ID))

print(f"=== Running Engine for Game {GAME_ID} ===")

# Load replay
php_snapshots = load_replay(str(ZIP_PATH))
envelopes = parse_p_envelopes_from_zip(ZIP_PATH)

# Get mapping from first snapshot
snap0 = php_snapshots[0]
co_p0, co_p1 = 9, 2  # from catalog
awbw_to_engine = map_snapshot_player_ids_to_engine(snap0, co_p0, co_p1)
print(f"Mapping: {awbw_to_engine}")

# Initialize engine
catalog = json.load(open('D:/awbw/data/amarriner_gl_std_catalog.json'))
game_info = catalog.get(str(GAME_ID), catalog.get(GAME_ID, {}))
map_id = game_info.get('maps_id', 159501)
m = load_map(map_id)
state = make_initial_state(m, co_p0, co_p1, first_mover=0)

print(f"Initial state created. Map: {map_id}")
print(f"P0 units: {len([u for u in state.units[0] if u.is_alive])}")
print(f"P1 units: {len([u for u in state.units[1] if u.is_alive])}")

# Run envelopes up to envelope 11 (before divergence)
for i, (pid, day, actions) in enumerate(envelopes[:12]):  # Run up to envelope 11
    for action in actions:
        state = apply_oracle_action_json(state, action, awbw_to_engine, envelope_awbw_player_id=pid)
    
    if i == 11:  # After envelope 11, about to compare to snapshot 12
        print(f"\nAfter envelope 11 (before snapshot 12):")
        print(f"  P0 units: {len([u for u in state.units[0] if u.is_alive])}")
        print(f"  P1 units: {len([u for u in state.units[1] if u.is_alive])}")
        
        # Check if engine has unit at P1 (6,7)
        print(f"\n  Checking engine for unit at P1 (6,7):")
        found = False
        for u in state.units[1]:
            if u.is_alive:
                r, c = u.pos
                if r == 6 and c == 7:
                    print(f"    FOUND: {u}")
                    found = True
        if not found:
            print(f"    NOT FOUND in engine")
            
        # Also check all P1 positions
        print(f"\n  All P1 unit positions:")
        for u in state.units[1]:
            if u.is_alive:
                r, c = u.pos
                print(f"    P1 ({r},{c}) {u.unit_type.name}")
