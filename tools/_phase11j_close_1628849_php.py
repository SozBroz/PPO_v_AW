#!/usr/bin/env python3
"""Compare PHP frames around env 25 of gid 1628849 to locate the missing 200g."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
)

GID = 1628849
ZIPS = ROOT / "replays" / "amarriner_gl"
zpath = ZIPS / f"{GID}.zip"
frames = load_replay(zpath)
envs = parse_p_envelopes_from_zip(zpath)

# Get awbw_to_engine via co ids
import json as _j
cat0 = ROOT / "data" / "amarriner_gl_std_catalog.json"
cat1 = ROOT / "data" / "amarriner_gl_extras_catalog.json"
by_id = {}
for cat in (cat0, cat1):
    if cat.exists():
        d = _j.loads(cat.read_text(encoding="utf-8"))
        for g in (d.get("games") or {}).values():
            if isinstance(g, dict) and "games_id" in g:
                by_id[int(g["games_id"])] = g
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
co0, co1 = pair_catalog_cos_ids(by_id[GID])
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}
print(f"awbw_to_engine={awbw_to_engine}")
print(f"engine P0 awbw={engine_to_awbw[0]} engine P1 awbw={engine_to_awbw[1]}")

# Print funds across env 23..26
print("\n=== Funds in PHP frames around env 25 ===")
print(f"{'frame_i':>8} {'day':>4} {'active':>10} {'P0_awbw':>8} {'P1_awbw':>8}")
for fi in range(22, 28):
    if fi >= len(frames):
        break
    f = frames[fi]
    funds = {}
    for pl in (f.get("players") or {}).values():
        try:
            funds[int(pl.get("id"))] = int(pl.get("funds") or 0)
        except (TypeError, ValueError):
            pass
    print(f"  {fi:>8} {f.get('day'):>4} {str(f.get('active_player_id')):>10} "
          f"{funds.get(engine_to_awbw[0]):>8} {funds.get(engine_to_awbw[1]):>8}")

# Inspect env 25 actions: identify what changes for P1 funds in PHP frames (from frame 25 -> 26)
print("\n=== Env 25 actions ===")
pid, day, actions = envs[25]
print(f"pid={pid} day={day} actions={len(actions)}")
for j, obj in enumerate(actions):
    kind = obj.get("action") or obj.get("type")
    extra = ""
    if kind == "Capt":
        bi = obj.get("buildingInfo", {})
        extra = f" cp_remain={bi.get('buildings_capture')} prev_team={bi.get('buildings_team')} pos=({bi.get('buildings_x')},{bi.get('buildings_y')}) income={obj.get('income')}"
    elif kind == "Build":
        nu = (obj.get("newUnit", {}) or {}).get("global", {})
        extra = f" unit={nu.get('units_name')} cost={nu.get('units_cost')} pos=({nu.get('units_x')},{nu.get('units_y')})"
    elif kind == "Fire":
        ci = obj.get("combatInfoVision", {}).get("global", {}).get("combatInfo", {})
        atk = ci.get("attacker", {})
        deff = ci.get("defender", {})
        extra = f" atk_id={atk.get('units_id')} pos=({atk.get('units_x')},{atk.get('units_y')}) hp={atk.get('units_hit_points')} -> def_id={deff.get('units_id')} pos=({deff.get('units_x')},{deff.get('units_y')}) hp={deff.get('units_hit_points')}"
    elif kind == "Power":
        extra = f" co={obj.get('coName')} power={obj.get('powerName')} type={obj.get('coPower')}"
    elif kind == "Move":
        path = obj.get("global") or []
        if path:
            extra = f" path={path[0]} -> {path[-1]} ({len(path)} steps)"
    print(f"  [{j:2}] {kind:10}{extra}")

# Compare frames 25 -> 26 for funds-relevant changes
print("\n=== Frame 25 vs 26 buildings (ownership changes) ===")
def buildings_dict(f):
    out = {}
    for b in (f.get("buildings") or {}).values():
        try:
            x = int(b.get("buildings_x"))
            y = int(b.get("buildings_y"))
            team = b.get("buildings_team")
            cap = b.get("buildings_capture")
            tid = b.get("terrain_id") or b.get("buildings_terrain_id")
            out[(x, y)] = (team, cap, tid)
        except (TypeError, ValueError):
            pass
    return out
b25 = buildings_dict(frames[25])
b26 = buildings_dict(frames[26])
diffs = 0
for k in set(b25.keys()) | set(b26.keys()):
    if b25.get(k) != b26.get(k):
        print(f"  {k}: 25={b25.get(k)}  -> 26={b26.get(k)}")
        diffs += 1
print(f"  total diffs: {diffs}")
