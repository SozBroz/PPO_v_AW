#!/usr/bin/env python3
"""Trace properties / unit at (15,3) before and after env 25 Capt action."""
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
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)

GID = 1628849
ZIPS = ROOT / "replays" / "amarriner_gl"
zpath = ZIPS / f"{GID}.zip"
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
map_data = load_map(int(meta["map_id"]), MAP_POOL := ROOT / "data" / "gl_map_pool.json", ROOT / "data" / "maps")
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

# Inspect properties at (15,3) and around
print(f"\n=== State at start of env 25 (after env 24 fully applied) ===")
print(f"P1 (Koal) funds: {state.funds[1]}")
print(f"P0 (Adder) funds: {state.funds[0]}")

# Map_data lookup uses (row, col). buildings_x=col, buildings_y=row.
# So (x=15, y=3) → (row=3, col=15)
target = (3, 15)
print(f"\n=== Property at (row=3, col=15) i.e. (x=15,y=3) ===")
prop = state.get_property_at(*target)
if prop:
    print(f"  owner={prop.owner} cp={prop.capture_points} terrain_id={prop.terrain_id}")
    print(f"  is_hq={prop.is_hq} is_base={prop.is_base} is_lab={prop.is_lab}")
    print(f"  is_comm_tower={prop.is_comm_tower} is_airport={prop.is_airport} is_port={prop.is_port}")
    print(f"  pos=({prop.row},{prop.col})")
else:
    print("  None at (3,15)")

# Find unit nearby
print("\nKoal P1 units near (3,15):")
for u in state.units[1]:
    dr = abs(u.pos[0] - 3); dc = abs(u.pos[1] - 15)
    if dr + dc <= 3:
        print(f"  {u.unit_type.name} at {u.pos} hp={u.hp} (display={(u.hp+9)//10}) capture_progress={u.capture_progress}")

print("\nAll Adder P0 properties (towers/income):")
for p in state.properties:
    if p.owner == 0:
        cls = "tower" if p.is_comm_tower else "lab" if p.is_lab else "income"
        if cls == "tower":
            print(f"  P0 TOWER at ({p.row},{p.col}) cp={p.capture_points} terrain={p.terrain_id}")

print("\nAll Koal P1 properties (towers/income):")
for p in state.properties:
    if p.owner == 1:
        cls = "tower" if p.is_comm_tower else "lab" if p.is_lab else "income"
        if cls == "tower":
            print(f"  P1 TOWER at ({p.row},{p.col}) cp={p.capture_points} terrain={p.terrain_id}")

# Now apply env 25 actions one by one and watch (15,3) and tower counts
print("\n=== Applying env 25 actions ===")
pid, day, actions = envs[25]
target_prop = state.get_property_at(*target)
for j, obj in enumerate(actions):
    kind = obj.get("action") or obj.get("type")
    p_before = state.get_property_at(*target)
    cp_b = p_before.capture_points if p_before else None
    own_b = p_before.owner if p_before else None
    p0_t = sum(1 for p in state.properties if p.owner == 0 and p.is_comm_tower)
    p1_t = sum(1 for p in state.properties if p.owner == 1 and p.is_comm_tower)
    f1_b = state.funds[1]
    try:
        apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
    except UnsupportedOracleAction as e:
        print(f"  [{j:2}] {kind:10} REFUSED: {e}")
        break
    p_after = state.get_property_at(*target)
    cp_a = p_after.capture_points if p_after else None
    own_a = p_after.owner if p_after else None
    p0_t2 = sum(1 for p in state.properties if p.owner == 0 and p.is_comm_tower)
    p1_t2 = sum(1 for p in state.properties if p.owner == 1 and p.is_comm_tower)
    f1_a = state.funds[1]
    df = f1_a - f1_b
    notes = []
    if (cp_b, own_b) != (cp_a, own_a):
        notes.append(f"(15,3) cp={cp_b}->{cp_a} own={own_b}->{own_a}")
    if (p0_t, p1_t) != (p0_t2, p1_t2):
        notes.append(f"P0_tow={p0_t}->{p0_t2} P1_tow={p1_t}->{p1_t2}")
    print(f"  [{j:2}] {kind:10} P1f={f1_a:>6} dP1={df:+6}  {'; '.join(notes)}")
