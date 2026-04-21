"""Replay 1631302 with hooks to trace Lander 192407337 and surface env=40 unload state."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.desync_audit import (
    CATALOG_DEFAULT, MAP_POOL_DEFAULT, MAPS_DIR_DEFAULT, ZIPS_DEFAULT,
    _load_catalog, pair_catalog_cos_ids,
)
from tools.oracle_zip_replay import (
    apply_oracle_action_json, parse_p_envelopes_from_zip,
    map_snapshot_player_ids_to_engine, resolve_replay_first_mover,
    load_replay,
)
from engine.map_loader import load_map
from engine.game import make_initial_state

GID = 1631302
LANDER_ID = 192407337
TARGET_ENV, TARGET_AI = 40, 10

cat = _load_catalog(CATALOG_DEFAULT)
games = cat.get('games') or {}
by_id = {int(g['games_id']): g for g in games.values() if isinstance(g, dict)}
meta = by_id[GID]
zip_path = ZIPS_DEFAULT / f'{GID}.zip'
co_p0, co_p1 = pair_catalog_cos_ids(meta)
map_id = int(meta['map_id'])
tier = str(meta.get('tier', 'T2'))

frames = load_replay(zip_path)
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
map_data = load_map(map_id, MAP_POOL_DEFAULT, MAPS_DIR_DEFAULT)
envelopes = parse_p_envelopes_from_zip(zip_path)
first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)
state = make_initial_state(map_data, co_p0, co_p1, starting_funds=0,
                           tier_name=tier or 'T2', replay_first_mover=first_mover)

def _all_units(state):
    """state.units is dict[player_idx -> list[Unit]]."""
    out = []
    if isinstance(state.units, dict):
        for ulist in state.units.values():
            out.extend(ulist)
    else:
        out.extend(state.units)
    return out

def find_lander_in_engine(state):
    """Find P0 Lander (the one with awbw_id=192407337 or, if unset, the only P0 LANDER alive)."""
    p0_landers = [u for u in (state.units.get(0) or []) if u.unit_type.name == 'LANDER' and u.hp > 0]
    if not p0_landers:
        return None
    matched = [u for u in p0_landers if getattr(u, 'awbw_units_id', None) == LANDER_ID]
    if matched:
        return matched[0]
    if len(p0_landers) > 1:
        print(f'!! WARNING: multiple P0 Landers: {[(u.pos, u.fuel, getattr(u,"awbw_units_id",None)) for u in p0_landers]}')
    return p0_landers[0]

prev_alive = True
prev_pos = None
prev_fuel = None
# Initial check
print(f'\n=== INITIAL UNIT INVENTORY ===')
for pidx, ulist in (state.units.items() if isinstance(state.units, dict) else []):
    for u in ulist:
        if u.unit_type.name == 'LANDER':
            print(f'  Lander: pos={u.pos} hp={u.hp} player={pidx} awbw_id={getattr(u,"awbw_units_id",None)} fuel={u.fuel}')

for ei, (pid, day, actions) in enumerate(envelopes):
    for ai, act in enumerate(actions):
        # Watch lander before and after this action — snapshot scalars
        _ld = find_lander_in_engine(state)
        ld_before_pos = _ld.pos if _ld else None
        ld_before_fuel = _ld.fuel if _ld else None
        ld_before_hp = _ld.hp if _ld else None
        ld_before = _ld  # for vanish detection (identity not needed)
        if ei == TARGET_ENV and ai == TARGET_AI:
            print(f'\n>>> About to apply env={ei} ai={ai} {act.get("action")}')
            print(f'    Engine state.active_player={state.active_player}, day={getattr(state,"day","?")}')
            lander = find_lander_in_engine(state)
            if lander is not None:
                print(f'    Lander 192407337 in engine: pos={lander.pos} fuel={lander.fuel} hp={lander.hp} player={lander.player_idx} loaded={[(c.unit_type.name, c.awbw_units_id) for c in lander.loaded_units]}')
            else:
                print(f'    Lander 192407337 NOT in engine state.units')
                if isinstance(state.units, dict):
                    for pidx, ulist in state.units.items():
                        for u in ulist:
                            if u.hp > 0 and u.unit_type.name == 'LANDER':
                                print(f'      candidate Lander: pos={u.pos} hp={u.hp} player={pidx} awbw_id={getattr(u,"awbw_units_id",None)} fuel={u.fuel}')
        try:
            apply_oracle_action_json(state, act, awbw_to_engine, envelope_awbw_player_id=pid)
            ld_after = find_lander_in_engine(state)
            # Detect vanish or significant fuel/pos change
            if ld_before is not None and ld_after is None:
                print(f'!! Lander VANISHED after env={ei} ai={ai} kind={act.get("action")} (was at {ld_before.pos} fuel={ld_before.fuel} hp={ld_before.hp})')
                # Dump full action
                print(json.dumps(act)[:400])
            elif ld_after is not None:
                if ld_before is not None and (ld_before_pos != ld_after.pos or ld_before_fuel != ld_after.fuel or ld_before_hp != ld_after.hp):
                    print(f'  env={ei} ai={ai} kind={act.get("action")}: Lander pos {ld_before_pos}->{ld_after.pos} fuel {ld_before_fuel}->{ld_after.fuel} hp {ld_before_hp}->{ld_after.hp}')
                if (prev_pos != ld_after.pos) or (prev_fuel != ld_after.fuel):
                    prev_pos = ld_after.pos
                    prev_fuel = ld_after.fuel
        except Exception as e:
            if ei == TARGET_ENV and ai == TARGET_AI:
                print(f'    EXCEPTION: {type(e).__name__}: {e}')
            print(f'STOP at env={ei} ai={ai}: {type(e).__name__}: {str(e)[:200]}')
            sys.exit(0)
print('completed all')
