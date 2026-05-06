#!/usr/bin/env python3
"""Test End action for game 1553655 directly"""
import sys
sys.path.insert(0, 'D:/awbw')

from pathlib import Path
from engine.game import make_initial_state
from engine.map_loader import load_map
from tools.oracle_zip_replay import (
    load_replay, parse_p_envelopes_from_zip,
    map_snapshot_player_ids_to_engine, resolve_replay_first_mover,
    apply_oracle_action_json,
)

# Setup
zip_path = Path('D:/awbw/replays/amarriner_gl/1553655.zip')
frames = load_replay(zip_path)
envelopes = parse_p_envelopes_from_zip(zip_path)
print(f'Loaded {len(frames)} frames, {len(envelopes)} envelopes')

# Create state
map_data = load_map(77060, Path('D:/awbw/data/gl_map_pool.json'), Path('D:/awbw/data/maps'))
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], 9, 2)
first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)
state = make_initial_state(
    map_data, 9, 2,
    starting_funds=0, tier_name='T2', replay_first_mover=first_mover
)
print(f'State created, active_player={state.active_player}')

# Apply envelope 0
pid, day, actions = envelopes[0]
print(f'Envelope 0: pid={pid}, day={day}, actions={len(actions)}')

for obj in actions:
    kind = obj.get('action')
    print(f'Applying {kind}...')
    try:
        apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
        print(f'  After {kind}: done={state.done}, active_player={state.active_player}')
    except Exception as e:
        print(f'  Exception on {kind}: {e}')
        import traceback
        traceback.print_exc()
        break

print('Test complete')
