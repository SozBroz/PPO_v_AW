"""Trace capture progress on (17,4) for 1631288 across envelopes 0-7."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay
from tools.desync_audit import (
    _run_replay_instrumented, _ReplayProgress,
    map_snapshot_player_ids_to_engine, parse_p_envelopes_from_zip,
    resolve_replay_first_mover, load_map,
)
from engine.game import make_initial_state

zp = Path('replays/amarriner_gl/1631288.zip')
frames = load_replay(zp)
co_p0, co_p1 = 11, 20
map_id = 159501

awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
map_data = load_map(map_id, Path('data/gl_map_pool.json'), Path('data/maps'))
envelopes = parse_p_envelopes_from_zip(zp)
first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)

target = (4, 17)  # row, col (engine convention is (row, col))

for stop_env in range(0, 8):
    state = make_initial_state(map_data, co_p0, co_p1, starting_funds=0, tier_name='T4', replay_first_mover=first_mover)
    progress = _ReplayProgress()
    truncated = envelopes[:stop_env+1]
    exc = _run_replay_instrumented(state, truncated, awbw_to_engine, progress, frames=frames, enable_state_mismatch=False, hp_internal_tolerance=0)
    prop = state.get_property_at(*target)
    if prop is None:
        print(f"env {stop_env}: no property at {target}")
        continue
    unit = state.get_unit_at(*target)
    udesc = f"{unit.unit_type.name} P{unit.player} hp={unit.hp}" if unit else "no unit"
    print(f"env {stop_env}: prop owner={prop.owner} cp={prop.capture_points} terrain={prop.terrain_id} | unit={udesc}")

# Now PHP frame 6 / 7 buildings at (17,4)
print()
for fi in (3, 4, 5, 6, 7, 8):
    f = frames[fi]
    bldgs = f.get('buildings') or {}
    if isinstance(bldgs, dict):
        bldgs = list(bldgs.values())
    php_b = next((b for b in bldgs if isinstance(b, dict) and b.get('x') == 17 and b.get('y') == 4), None)
    if php_b:
        print(f"PHP f{fi}: ({php_b.get('x')},{php_b.get('y')}) terrain={php_b.get('terrain_id')} cap={php_b.get('capture')} last_capture={php_b.get('last_capture')}")
    units_php = f.get('units') or {}
    if isinstance(units_php, dict):
        units_php = list(units_php.values())
    php_u = [u for u in units_php if isinstance(u, dict) and u.get('x') == 17 and u.get('y') == 4]
    for u in php_u:
        print(f"  PHP unit at (17,4): id={u.get('id')} pid={u.get('players_id')} type={u.get('name') or u.get('units_name')} hp={u.get('hit_points')}")

# Also: dump env 6 actions (to inspect what P0 did on day 4)
print()
print("--- env 6 actions ---")
import json
for ei in (4, 5, 6):
    env = envelopes[ei]
    print(f"\nenv {ei} type={type(env).__name__}")
    # Try to dump structure
    if isinstance(env, tuple):
        for j, item in enumerate(env):
            s = json.dumps(item, default=str)[:600] if not isinstance(item, str) else item[:200]
            print(f"  field {j} ({type(item).__name__}): {s}")
    else:
        print(f"  raw: {json.dumps(env, default=str)[:1000]}")
