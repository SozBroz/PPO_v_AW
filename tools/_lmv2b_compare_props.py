"""Compare PHP frame 7 Sami-owned props vs engine post-env-7 Sami-owned props for 1631288."""
import sys
from pathlib import Path
from collections import defaultdict

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
print("awbw_to_engine:", awbw_to_engine)
map_data = load_map(map_id, Path('data/gl_map_pool.json'), Path('data/maps'))
envelopes = parse_p_envelopes_from_zip(zp)
first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)

# Compare ENGINE state at end of envelope N vs PHP frame N+1 for N in 0..7
for stop_env in range(0, 8):
    state = make_initial_state(map_data, co_p0, co_p1, starting_funds=0, tier_name='T4', replay_first_mover=first_mover)
    progress = _ReplayProgress()
    truncated = envelopes[:stop_env+1]
    exc = _run_replay_instrumented(state, truncated, awbw_to_engine, progress, frames=frames, enable_state_mismatch=False, hp_internal_tolerance=0)
    if exc is not None:
        print(f"env {stop_env}: exc={exc}")
        break
    eng_p0_props = sorted([(p.col, p.row, p.terrain_id) for p in state.properties if p.owner == 0])

    # PHP frame stop_env+1
    php_frame = frames[stop_env + 1]
    bldgs = php_frame.get('buildings') or {}
    if isinstance(bldgs, dict):
        bldgs = list(bldgs.values())
    # Buildings don't have explicit ownership in dict; encoded by terrain_id range.
    # countries: BM=2 (43-47), BD=8 (96-100). Sami is countries_id=2.
    sami_terrain_range = set(range(43, 48))  # BM city/base/airport/port/HQ
    php_p0_props = sorted([(b.get('x'), b.get('y'), b.get('terrain_id')) for b in bldgs if isinstance(b, dict) and (b.get('terrain_id') or 0) in sami_terrain_range])

    eng_set = {(c, r) for c, r, t in eng_p0_props}
    php_set = {(c, r) for c, r, t in php_p0_props}
    only_eng = eng_set - php_set
    only_php = php_set - eng_set

    funds_eng = state.funds
    php_funds = {p.get('id'): p.get('funds') for p in (php_frame.get('players', {}).values() if isinstance(php_frame.get('players'), dict) else php_frame.get('players', []))}

    print(f"env {stop_env} | engine_funds=P0={funds_eng[0]} P1={funds_eng[1]} | php_funds={php_funds}")
    print(f"           eng P0_props={len(eng_p0_props)} php P0_props={len(php_p0_props)} | only_eng={only_eng} | only_php={only_php}")
