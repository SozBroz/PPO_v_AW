#!/usr/bin/env python3
"""Test what happens when we apply the End action for game 1553655"""
import sys
sys.path.insert(0, 'D:/awbw')

from pathlib import Path
from engine.game import make_initial_state
from engine.map_loader import load_map
from tools.oracle_zip_replay import (
    load_replay, parse_p_envelopes_from_zip,
    map_snapshot_player_ids_to_engine, resolve_replay_first_mover,
    apply_oracle_action_json
)

# Load game 1553655
zip_path = Path('D:/awbw/replays/amarriner_gl/1553655.zip')
frames = load_replay(zip_path)
print(f'Loaded {len(frames)} frames')

# Parse envelopes
envelopes = parse_p_envelopes_from_zip(zip_path)
print(f'Parsed {len(envelopes)} envelopes')

# Get CO ids from catalog
import json
catalog = json.load(open('D:/awbw/data/amarriner_gl_std_catalog.json'))
meta = None
for g in catalog['games']:
    if int(g['games_id']) == 1553655:
        meta = g
        break

co_p0, co_p1 = int(meta['co_p0_id']), int(meta['co_p1_id'])
print(f'COs: p0={co_p0}, p1={co_p1}')

# Create initial state
map_data = load_map(77060, None, Path('D:/awbw/data/maps'))
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)

state = make_initial_state(
    map_data,
    co_p0,
    co_p1,
    starting_funds=0,
    tier_name='T2',
    replay_first_mover=first_mover,
)
print(f'State created, active_player={state.active_player}')

# Apply first envelope (envelope 0)
pid, day, actions = envelopes[0]
print(f'Envelope 0: pid={pid}, day={day}, n_actions={len(actions)}')

for obj in actions:
    action_kind = obj.get('action')
    print(f'Applying action: {action_kind}')
    try:
        apply_oracle_action_json(
            state, obj, awbw_to_engine,
            envelope_awbw_player_id=pid
        )
        if action_kind == 'End':
            print(f'After End: active_player={state.active_player}, done={state.done}')
    except Exception as e:
        print(f'Exception on action {action_kind}: {e}')
        import traceback
        traceback.print_exc()
        break

print('Test complete')
