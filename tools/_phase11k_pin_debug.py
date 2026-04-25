#!/usr/bin/env python3
"""Verify the post-envelope pin actually changes engine HP at envelope 10
of gid 1635679 — RECON id 192721109 should drop from 100 to 97 internal."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import make_initial_state
from engine.map_loader import load_map
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.replay_snapshot_compare import replay_snapshot_pairing

CATS = [ROOT / "data" / "amarriner_gl_extras_catalog.json"]
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"

GID = 1635679

by_id = {}
for cp in CATS:
    cat = json.loads(cp.read_text(encoding="utf-8"))
    for g in (cat.get("games") or {}).values():
        if isinstance(g, dict) and "games_id" in g:
            by_id[int(g["games_id"])] = g
meta = by_id[GID]
random.seed(_seed_for_game(CANONICAL_SEED, GID))
co0, co1 = pair_catalog_cos_ids(meta)
map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
zpath = ZIPS / f"{GID}.zip"
envs = parse_p_envelopes_from_zip(zpath)
frames = load_replay(zpath)
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
state = make_initial_state(map_data, co0, co1, starting_funds=0,
                           tier_name=str(meta.get("tier") or "T2"),
                           replay_first_mover=first_mover)

for env_i, (pid, day, actions) in enumerate(envs):
    pf = frames[env_i + 1]
    pin = {}
    for u in (pf.get("units") or {}).values():
        try:
            pin[int(u["id"])] = max(0, min(100, int(round(float(u["hit_points"]) * 10))))
        except (TypeError, ValueError, KeyError):
            continue
    state._oracle_post_envelope_units_by_id = pin
    if env_i == 10:
        print(f"env=10 pin contains {len(pin)} units; recon id 192721109 internal={pin.get(192721109)}")
    for obj in actions:
        apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
    if env_i == 10:
        u = next((uu for uu in state.units[0] if uu.unit_id and (3, 4) == uu.pos), None)
        if u:
            print(f"AFTER env=10: engine recon at (3,4) hp={u.hp} (expected 97 with pin)")
        else:
            for uu in state.units[0]:
                if uu.pos == (3, 4):
                    print(f"AFTER env=10: engine unit at (3,4) {uu.unit_type.name} hp={uu.hp}")
        break
state._oracle_post_envelope_units_by_id = None
