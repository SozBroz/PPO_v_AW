#!/usr/bin/env python3
"""Dump legal actions when processing env 25 action [3] Join in engine."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from engine.action import get_legal_actions

GID = 1628849
zpath = ROOT / "replays" / "amarriner_gl" / f"{GID}.zip"
envs = parse_p_envelopes_from_zip(zpath)
frames = load_replay(zpath)

cat0 = ROOT / "data" / "amarriner_gl_std_catalog.json"
cat1 = ROOT / "data" / "amarriner_gl_extras_catalog.json"
by_id = {}
for cat in (cat0, cat1):
    if cat.exists():
        d = json.loads(cat.read_text(encoding="utf-8"))
        for g in (d.get("games") or {}).values():
            if isinstance(g, dict) and "games_id" in g:
                by_id[int(g["games_id"])] = g
meta = by_id[GID]
random.seed(_seed_for_game(CANONICAL_SEED, GID))
co0, co1 = pair_catalog_cos_ids(meta)
map_data = load_map(int(meta["map_id"]), ROOT / "data" / "gl_map_pool.json", ROOT / "data" / "maps")
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

state = make_initial_state(map_data, co0, co1, starting_funds=0,
                            tier_name=str(meta.get("tier") or "T2"),
                            replay_first_mover=first_mover)

# Apply through env 24
for env_i in range(25):
    pid, day, actions = envs[env_i]
    for obj in actions:
        apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)

pid, day, actions = envs[25]
# Apply [0] Power, [1] Fire, [2] Capt — instrumented like full trace
import tools.oracle_zip_replay as ozr
orig_step_pre = ozr._engine_step
def pre_step(state, action, hook):
    return orig_step_pre(state, action, hook)
ozr._engine_step = pre_step
for j in (0, 1, 2):
    apply_oracle_action_json(state, actions[j], awbw_to_engine, envelope_awbw_player_id=pid)
ozr._engine_step = orig_step_pre

print("=== After actions [0..2] ===")
print(f"  prop (3,15): {state.get_property_at(3,15).owner=}, cp={state.get_property_at(3,15).capture_points}")
print("  Koal P1 units near (3,15):")
for u in state.units[1]:
    if abs(u.pos[0]-3)+abs(u.pos[1]-15) <= 6:
        print(f"    {u.unit_type.name} at {u.pos} hp={u.hp} display={(u.hp+9)//10} unit_id={getattr(u,'unit_id',None)} fuel={u.fuel} ammo={u.ammo} cap_prog={getattr(u,'capture_progress',None)}")

# Now look at what action [3] Join would do — dump pre-state
print("\n=== Action [3] Join JSON snippet ===")
move = actions[3].get("Move", {})
paths = (move.get("paths") or {}).get("global") or []
print(f"  paths.global: {paths}")
unit = move.get("unit", {})
g = unit.get("global", {})
print(f"  mover unit_id={g.get('units_id')} units_x={g.get('units_x')} units_y={g.get('units_y')} hp={g.get('units_hit_points')}")

# Apply [3] Join with monkey-patched logging
from engine.action import ActionType
import tools.oracle_zip_replay as ozr
orig_step = ozr._engine_step
def patched_step(state, action, hook):
    print(f"  _engine_step: action_type={action.action_type.name} unit_pos={getattr(action,'unit_pos',None)} move_pos={getattr(action,'move_pos',None)} unit_type={getattr(action,'unit_type',None)} select_id={getattr(action,'select_unit_id',None)}")
    print(f"    state: active_player={state.active_player} stage={state.action_stage.name} selected_unit={state.selected_unit and (state.selected_unit.unit_type.name, state.selected_unit.pos, state.selected_unit.hp)}")
    return orig_step(state, action, hook)
ozr._engine_step = patched_step
print("\n=== Applying [3] Join... ===")
try:
    apply_oracle_action_json(state, actions[3], awbw_to_engine, envelope_awbw_player_id=pid)
except Exception as e:
    print(f"  EXC: {type(e).__name__}: {e}")
ozr._engine_step = orig_step
print(f"  AFTER: prop (3,15) owner={state.get_property_at(3,15).owner} cp={state.get_property_at(3,15).capture_points}")
print("  Koal P1 units near (3,15):")
for u in state.units[1]:
    if abs(u.pos[0]-3)+abs(u.pos[1]-15) <= 6:
        print(f"    {u.unit_type.name} at {u.pos} hp={u.hp} display={(u.hp+9)//10} unit_id={getattr(u,'unit_id',None)} cap_prog={getattr(u,'capture_progress',None)}")
print(f"  P0 funds={state.funds[0]} P1 funds={state.funds[1]}")
