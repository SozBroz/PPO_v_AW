#!/usr/bin/env python3
"""After applying envelopes 0..24 with the post-frame pin enabled, dump
engine vs PHP HP for the Sturm units we care about: TANK (13,13),
INFANTRY (17,15), INFANTRY (7,9), and also dump every Sturm Fire from
env 24 with attacker pos / pin / engine post-counter HP so we see
whether the pin is actually changing counter damage on this turn."""
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

CATS = [Path("data/amarriner_gl_extras_catalog.json")]
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
GID = 1635679
TARGET_POSITIONS = {(13, 13), (17, 15), (7, 9)}

by_id = {}
for cp in CATS:
    cat = json.loads((ROOT / cp).read_text(encoding="utf-8"))
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
    pf = frames[env_i + 1] if env_i + 1 < len(frames) else None
    pin = {}
    if pf is not None:
        for u in (pf.get("units") or {}).values():
            try:
                pin[int(u["id"])] = max(0, min(100, int(round(float(u["hit_points"]) * 10))))
            except (TypeError, ValueError, KeyError):
                continue
    state._oracle_post_envelope_units_by_id = pin
    if env_i == 24:
        print(f"--- env=24 Sturm day=13 fires (pin={len(pin)} entries) ---")
    for ai, obj in enumerate(actions):
        kind = obj.get("action") if isinstance(obj, dict) else None
        if env_i == 24 and kind == "Fire":
            ci = (obj.get("Fire") or {}).get("combatInfoVision") or {}
            ci_g = (ci.get("global") or {}).get("combatInfo") or {}
            att = ci_g.get("attacker") or {}
            df = ci_g.get("defender") or {}
            apos = (att.get("units_y"), att.get("units_x"))
            dpos = (df.get("units_y"), df.get("units_x"))
            att_uid = att.get("units_id")
            try:
                att_uid_i = int(att_uid)
            except (TypeError, ValueError):
                att_uid_i = None
            pre = next((u for u in state.units[0] if u.unit_id and u.pos == (apos[0] or -1, apos[1] or -1)), None)
            apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
            post = None
            for player_seat in (0, 1):
                for u in state.units[player_seat]:
                    if u.unit_id is not None and u.is_alive and u.pos == (apos[0] or -1, apos[1] or -1):
                        post = u
                        break
                if post:
                    break
            print(f"  ai={ai} att_pos={apos} att_uid={att_uid} ci_att_hp={att.get('units_hit_points')} "
                  f"pin_att_hp={pin.get(att_uid_i)}  -> engine post hp={post.hp if post else 'gone'}")
        else:
            apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
    if env_i == 24:
        print()
        for tgt in TARGET_POSITIONS:
            for u in state.units[0]:
                if u.pos == tgt:
                    php_hp = None
                    pf25 = frames[25]
                    for pu in (pf25.get("units") or {}).values():
                        if (int(pu["y"]), int(pu["x"])) == tgt and int(pu["players_id"]) == 3778256:
                            php_hp = pu["hit_points"]
                            break
                    print(f"=== POST-env24 ({u.unit_type.name} at {tgt}): engine hp={u.hp}  PHP hp={php_hp}  ===")
                    break
        break
state._oracle_post_envelope_units_by_id = None
