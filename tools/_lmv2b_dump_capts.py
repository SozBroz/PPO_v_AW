"""Dump every Capt action across envelopes and the engine's resulting (4,17) state."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay
from tools.desync_audit import (
    parse_p_envelopes_from_zip, _ReplayProgress, _run_replay_instrumented,
    map_snapshot_player_ids_to_engine, resolve_replay_first_mover, load_map,
)
from engine.game import make_initial_state

zp = Path('replays/amarriner_gl/1631288.zip')
frames = load_replay(zp)
envelopes = parse_p_envelopes_from_zip(zp)

# Dump every Capt action across envelopes
for ei, env in enumerate(envelopes):
    if not isinstance(env, tuple) or len(env) < 3:
        continue
    pid, day, actions = env
    if not isinstance(actions, list):
        continue
    for ai, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        if a.get('action') == 'Capt':
            capt = a.get('Capt', {}) or {}
            move = a.get('Move', {}) or {}
            bi = capt.get('buildingInfo', {}) or {}
            x = bi.get('buildings_x')
            y = bi.get('buildings_y')
            cp = bi.get('buildings_capture')
            owner = bi.get('buildings_players_id')
            # Get unit position from Move (the Move action moves unit to target tile)
            mu = move.get('unit') if isinstance(move, dict) else None
            mdest = None
            if isinstance(mu, dict):
                # nested dict
                for v in mu.values():
                    if isinstance(v, dict):
                        for v2 in v.values():
                            if isinstance(v2, dict) and 'units_x' in v2:
                                mdest = (v2.get('units_x'), v2.get('units_y'))
                                break
                        if mdest is None and 'units_x' in v:
                            mdest = (v.get('units_x'), v.get('units_y'))
                    if mdest:
                        break
            mdist = move.get('dist') if isinstance(move, dict) else None
            print(f"env {ei} (pid={pid} day={day}) action {ai} Capt at building ({x},{y}) cp={cp} owner={owner} | unit dest={mdest}")

# Compare engine state at end of each env vs PHP frame at start of next env at (4,17)
print("\n=== Engine env-end vs PHP frame-start at (col=17,row=4) ===")
co_p0, co_p1 = 11, 20
map_id = 159501
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
map_data = load_map(map_id, Path('data/gl_map_pool.json'), Path('data/maps'))
first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)

for stop_env in range(0, 10):
    state = make_initial_state(map_data, co_p0, co_p1, starting_funds=0, tier_name='T4', replay_first_mover=first_mover)
    progress = _ReplayProgress()
    truncated = envelopes[:stop_env+1]
    _run_replay_instrumented(state, truncated, awbw_to_engine, progress, frames=frames, enable_state_mismatch=False, hp_internal_tolerance=0)
    prop = state.get_property_at(4, 17)
    unit = state.get_unit_at(4, 17)
    udesc = f"{unit.unit_type.name} P{unit.player} hp={unit.hp} uid={unit.unit_id}" if unit else "no unit"
    eng_desc = f"prop owner={prop.owner if prop else '-'} cp={prop.capture_points if prop else '-'} terr={prop.terrain_id if prop else '-'}"

    fi = stop_env + 1
    if fi < len(frames):
        f = frames[fi]
        bldgs = f.get('buildings') or {}
        if isinstance(bldgs, dict):
            bldgs = list(bldgs.values())
        php_b = next((b for b in bldgs if isinstance(b, dict) and b.get('x') == 17 and b.get('y') == 4), None)
        units_php = f.get('units') or {}
        if isinstance(units_php, dict):
            units_php = list(units_php.values())
        php_u = next((u for u in units_php if isinstance(u, dict) and u.get('x') == 17 and u.get('y') == 4), None)
        php_desc = f"prop terr={php_b.get('terrain_id') if php_b else '-'} cap={php_b.get('capture') if php_b else '-'}"
        php_udesc = f"{php_u.get('name') or php_u.get('units_name')} pid={php_u.get('players_id')} hp={php_u.get('hit_points')} uid={php_u.get('id')}" if php_u else "no unit"
    else:
        php_desc = "no frame"
        php_udesc = "-"

    print(f"env {stop_env} → f{fi}:")
    print(f"  engine: {eng_desc} | unit={udesc}")
    print(f"  php   : {php_desc} | unit={php_udesc}")
