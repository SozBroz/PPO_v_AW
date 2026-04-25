"""Trace property (4,17) cp before/after each action in env 4.

Re-runs the audit envelope-by-envelope, but for env 4 sub-applies actions one at a time using oracle_zip_replay.apply_p_envelope_with_progress.
"""
import sys, json, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay
from tools.desync_audit import (
    parse_p_envelopes_from_zip, _ReplayProgress, _run_replay_instrumented,
    map_snapshot_player_ids_to_engine, resolve_replay_first_mover, load_map,
)
from engine.game import make_initial_state
from tools import oracle_zip_replay as ozr

zp = Path('replays/amarriner_gl/1631288.zip')
frames = load_replay(zp)
envelopes = parse_p_envelopes_from_zip(zp)
co_p0, co_p1 = 11, 20
map_id = 159501
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
map_data = load_map(map_id, Path('data/gl_map_pool.json'), Path('data/maps'))
first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)

state = make_initial_state(map_data, co_p0, co_p1, starting_funds=0, tier_name='T4', replay_first_mover=first_mover)
progress = _ReplayProgress()
# Run envelopes 0..3 first (P1 day 1, P0 day 2, P1 day 2, P0 day 2-end... actually just 0-3)
truncated = envelopes[:4]
_run_replay_instrumented(state, truncated, awbw_to_engine, progress, frames=frames, enable_state_mismatch=False, hp_internal_tolerance=0)
prop = state.get_property_at(4, 17)
print(f"Before env 4: prop cp={prop.capture_points} owner={prop.owner} terr={prop.terrain_id}")
unit = state.get_unit_at(4, 17)
print(f"  unit at (4,17) before env 4: {unit}")

# Now process env 4 actions one at a time
env4 = envelopes[4]
pid_awbw, day, actions = env4
engine_pid = awbw_to_engine.get(pid_awbw, pid_awbw)
print(f"\nProcessing env 4: awbw_pid={pid_awbw}->engine_p{engine_pid} day={day}")

for ai, action in enumerate(actions):
    # Build a single-action envelope and apply
    print(f"\n--- action {ai} kind={action.get('action') if isinstance(action, dict) else type(action).__name__} ---")
    try:
        ozr.apply_oracle_action_json(state, action, awbw_to_engine, envelope_awbw_player_id=pid_awbw)
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {e}")
    prop = state.get_property_at(4, 17)
    unit = state.get_unit_at(4, 17)
    print(f"  AFTER (4,17): prop cp={prop.capture_points} owner={prop.owner} terr={prop.terrain_id} | unit={unit}")
