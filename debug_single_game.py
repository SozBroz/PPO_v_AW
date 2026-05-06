"""Debug - run single game audit with detailed logging."""
import sys
sys.path.insert(0, 'D:/awbw')

from tools.oracle_zip_replay import load_replay, parse_p_envelopes_from_zip, resolve_replay_first_mover, map_snapshot_player_ids_to_engine
from tools.replay_snapshot_compare import compare_snapshot_to_engine
from engine.game import GameState

GAME_ID = 1553655
ZIP_PATH = f'D:/awbw/replays/amarriner_gl/{GAME_ID}.zip'

print(f"=== Auditing Single Game {GAME_ID} ===")

# Load replay
php_snapshots = load_replay(ZIP_PATH)
envelopes = parse_p_envelopes_from_zip(ZIP_PATH)
print(f"PHP snapshots: {len(php_snapshots)}, Envelopes: {len(envelopes)}")

# Get mapping from first snapshot
snap0 = php_snapshots[0]
co_p0, co_p1 = 9, 2  # from catalog
awbw_to_engine = map_snapshot_player_ids_to_engine(snap0, co_p0, co_p1)
print(f"Mapping: {awbw_to_engine}")

# Determine pairing mode
n_frames = len(php_snapshots)
n_envelopes = len(envelopes)
print(f"Frames={n_frames}, Envelopes={n_envelopes}")

# Trailing: N frames for N-1 envelopes (frame[i+1] after envelope i)
# Tight: N frames for N envelopes
if n_frames == n_envelopes + 1:
    pairing = "trailing"
elif n_frames == n_envelopes:
    pairing = "tight"
else:
    pairing = "unknown"
print(f"Pairing mode: {pairing}")

# Run envelopes and compare
state = None
for i, (pid, day, actions) in enumerate(envelopes):
    # Apply envelope
    from tools.oracle_zip_replay import apply_oracle_action_json
    if state is None:
        from engine.game import make_initial_state
        from data.maps import load_map
        import json
        catalog = json.load(open('D:/awbw/data/amarriner_gl_std_catalog.json'))
        game_info = catalog.get(str(GAME_ID), catalog.get(GAME_ID, {}))
        map_id = game_info.get('maps_id', 159501)
        from data.maps import load_map
        m = load_map(map_id)
        state = make_initial_state(m, co_p0, co_p1, first_mover=0)
    
    for action in actions:
        state = apply_oracle_action_json(state, action, awbw_to_engine, envelope_awbw_player_id=pid)
    
    # Compare to snapshot i+1
    snap_idx = i + 1
    if snap_idx < n_frames:
        php = php_snapshots[snap_idx]
        
        # Run compare_snapshot_to_engine manually
        from tools.replay_snapshot_compare import (
            compare_funds, compare_units, compare_properties, 
            compare_co_states, compare_weather, compare_turn
        )
        
        results = []
        results.extend(compare_funds(php, state, awbw_to_engine))
        results.extend(compare_units(php, state, awbw_to_engine))
        results.extend(compare_properties(php, state, awbw_to_engine))
        results.extend(compare_co_states(php, state, awbw_to_engine))
        results.extend(compare_weather(php, state))
        results.extend(compare_turn(php, state))
        
        if results:
            print(f"\nDivergence after envelope {i} (snapshot {snap_idx}):")
            for r in results[:5]:
                print(f"  {r}")
            
            # Check specifically for (1,6,7)
            for r in results:
                if '6, 7' in r or '(1, 6, 7)' in r:
                    print(f"\n*** FOUND (1,6,7) MISMATCH AT ENVELOPE {i} ***")
                    print(f"  Full message: {r}")
                    # Print PHP units at (6,7)
                    units = php.get('units', {})
                    for k, u in (units if isinstance(units, dict) else {}).items():
                        if not isinstance(u, dict): continue
                        col, row = int(u['x']), int(u['y'])
                        if row == 6 and col == 7:
                            print(f"  PHP unit at (6,7): {u}")
                    break
            if i > 15:  # Stop after envelope 15
                break
    else:
        break  # No more snapshots to compare

print(f"\nDone.")
