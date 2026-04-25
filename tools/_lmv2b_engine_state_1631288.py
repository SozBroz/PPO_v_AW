"""Run 1631288 in-process and inspect engine property ownership at frame 8 (start of P0 day 5)."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.diff_replay_zips import load_replay
from tools.desync_audit import (
    _run_replay_instrumented,
    _ReplayProgress,
    map_snapshot_player_ids_to_engine,
    pair_catalog_cos_ids,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
    load_map,
)
from engine.game import make_initial_state, GameState

zp = Path('replays/amarriner_gl/1631288.zip')
frames = load_replay(zp)
print(f"frames: {len(frames)}")

co_p0, co_p1 = 11, 20
map_id = 159501
meta = {'co_p0_id': co_p0, 'co_p1_id': co_p1, 'map_id': map_id}

awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
map_data = load_map(map_id, Path('data/gl_map_pool.json'), Path('data/maps'))
envelopes = parse_p_envelopes_from_zip(zp)
first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)
state = make_initial_state(map_data, co_p0, co_p1, starting_funds=0, tier_name='T4', replay_first_mover=first_mover)

# Capture state snapshots after each envelope
import tools.desync_audit as da
orig_run = da._run_replay_instrumented

snaps = []

# Monkey-patch the loop. Easier: copy state at end of envelope 7 by hooking via short-circuit
# Quick approach: just run all envelopes but stop after envelope 7 by truncating envelopes
truncated = envelopes[:8]  # envelopes 0..7

progress = _ReplayProgress()
exc = orig_run(state, truncated, awbw_to_engine, progress, frames=frames, enable_state_mismatch=False, hp_internal_tolerance=0)
print(f"exc: {exc}")
print(f"envelopes_applied: {progress.envelopes_applied} actions: {progress.actions_applied}")
print(f"engine funds: P0={state.funds[0]} P1={state.funds[1]}")
print(f"engine current_player={state.active_player} day(turn)={state.turn}")

# Property ownership
by_owner_terrain = defaultdict(lambda: defaultdict(int))
for prop in state.properties:
    owner_key = 'neutral' if prop.owner is None else prop.owner
    by_owner_terrain[owner_key][prop.terrain_id] += 1
for owner, terrains in by_owner_terrain.items():
    print(f"  owner={owner}: {dict(terrains)} (total={sum(terrains.values())})")

# Income props for P0 (Sami)
p0_income_props = [p for p in state.properties if p.owner == 0 and not p.is_lab and not p.is_comm_tower]
print(f"P0 (Sami) income properties: {len(p0_income_props)}")
for p in p0_income_props:
    print(f"  ({p.col},{p.row}) terrain={p.terrain_id} hq={p.is_hq} base={p.is_base} city={not (p.is_hq or p.is_base or p.is_airport or p.is_port or p.is_comm_tower or p.is_lab)} airport={p.is_airport} port={p.is_port}")

# Now compare to PHP frame 7 (just before P0 day 5 income)
# Actually frames[8] is "start of P0 day 5 AFTER income" per our earlier analysis.
# Let me dump PHP frame 7 (start of P1 day 4) buildings owned by Sami's seat (awbw_id 3769259)
sami_pid = 3769259
f7 = frames[7]
bldgs7 = f7.get('buildings') or {}
if isinstance(bldgs7, dict):
    bldgs7 = list(bldgs7.values())
sami_props_php_f7 = [b for b in bldgs7 if isinstance(b, dict) and b.get('players_id') == sami_pid]
print(f"PHP f7 (start P1 day 4): Sami owns {len(sami_props_php_f7)} buildings")
for b in sami_props_php_f7:
    print(f"  PHP ({b.get('x')},{b.get('y')}) terrain={b.get('terrain_id')} pid={b.get('players_id')} cap={b.get('capture')} last_capture={b.get('last_capture')}")
