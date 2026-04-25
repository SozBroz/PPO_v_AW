#!/usr/bin/env python3
"""Inspect HP / unit state at frame 25 vs 26 + check capture completions in env 24-25."""
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
from tools.amarriner_catalog_cos import pair_catalog_cos_ids

GID = 1628849
ZIPS = ROOT / "replays" / "amarriner_gl"
zpath = ZIPS / f"{GID}.zip"
frames = load_replay(zpath)
envs = parse_p_envelopes_from_zip(zpath)

cat0 = ROOT / "data" / "amarriner_gl_std_catalog.json"
cat1 = ROOT / "data" / "amarriner_gl_extras_catalog.json"
by_id = {}
for cat in (cat0, cat1):
    if cat.exists():
        d = json.loads(cat.read_text(encoding="utf-8"))
        for g in (d.get("games") or {}).values():
            if isinstance(g, dict) and "games_id" in g:
                by_id[int(g["games_id"])] = g
co0, co1 = pair_catalog_cos_ids(by_id[GID])
awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}

# Look at units in frames 24, 25, 26 belonging to P1 (Koal awbw=3763927) on properties
print(f"Engine P1 awbw_id = {engine_to_awbw[1]}")

def units_dict(f):
    out = {}
    for u in (f.get("units") or {}).values():
        try:
            uid = int(u.get("units_id"))
            out[uid] = {
                "type": u.get("units_name"),
                "x": int(u.get("units_x")),
                "y": int(u.get("units_y")),
                "hp": int(u.get("units_hit_points") or 0),
                "player": int(u.get("units_players_id") or 0),
                "moved": u.get("units_moved"),
                "fuel": int(u.get("units_fuel") or 0),
                "ammo": int(u.get("units_ammo") or 0),
                "cap": int(u.get("units_capture") or 0),
            }
        except (TypeError, ValueError):
            pass
    return out

p1_awbw = engine_to_awbw[1]
print("\n=== P1 (Koal) units across frames 23..27 (HP changes) ===")
prev = None
all_uids = set()
for fi in range(23, 28):
    if fi >= len(frames):
        break
    u = units_dict(frames[fi])
    p1 = {uid: d for uid, d in u.items() if d["player"] == p1_awbw}
    all_uids |= set(p1.keys())

# Print HP table
print(f"{'unit_id':>10} {'type':>10} ", end="")
for fi in range(23, 28):
    print(f" f{fi}_hp f{fi}_pos      ", end="")
print()

uref = units_dict(frames[25])  # reference frame
for uid in sorted(all_uids):
    info = uref.get(uid) or {}
    print(f"{uid:>10} {info.get('type','?')[:10]:>10} ", end="")
    for fi in range(23, 28):
        if fi >= len(frames):
            print(f" {'-':>5}", end="")
            continue
        u = units_dict(frames[fi]).get(uid)
        if u and u["player"] == p1_awbw:
            print(f"  {u['hp']:>3}  ({u['x']:>2},{u['y']:>2})    ", end="")
        else:
            print(f"  ---  --------     ", end="")
    print()

# Capture completions: look at building ownership transitions in env 24 -> env 25
print("\n=== Building ownership transitions ===")
def b_dict(f):
    out = {}
    for b in (f.get("buildings") or {}).values():
        try:
            x = int(b.get("buildings_x")); y = int(b.get("buildings_y"))
            tm = b.get("buildings_team")
            cap = b.get("buildings_capture")
            out[(x,y)] = (str(tm) if tm not in (None, "") else None, cap)
        except (TypeError, ValueError):
            pass
    return out
for fi in range(23, 27):
    if fi+1 >= len(frames):
        break
    b1 = b_dict(frames[fi])
    b2 = b_dict(frames[fi+1])
    diffs = []
    for k in set(b1.keys()) | set(b2.keys()):
        if b1.get(k) != b2.get(k):
            diffs.append((k, b1.get(k), b2.get(k)))
    if diffs:
        print(f" frame {fi} -> {fi+1}:")
        for k, a, b in diffs:
            print(f"   {k}: {a} -> {b}")

# Print env 24 actions briefly
print("\n=== Env 24 action summary ===")
pid, day, actions = envs[24]
print(f"pid={pid} day={day} n={len(actions)}")
for j, obj in enumerate(actions):
    kind = obj.get("action") or obj.get("type")
    extra = ""
    if kind == "Capt":
        bi = obj.get("buildingInfo", {})
        extra = f" cp={bi.get('buildings_capture')} prev_team={bi.get('buildings_team')} pos=({bi.get('buildings_x')},{bi.get('buildings_y')}) income={obj.get('income')}"
    elif kind == "Build":
        nu = (obj.get("newUnit", {}) or {}).get("global", {})
        extra = f" unit={nu.get('units_name')} cost={nu.get('units_cost')}"
    elif kind == "Power":
        extra = f" co={obj.get('coName')} power={obj.get('powerName')}"
    print(f"  [{j:2}] {kind:10}{extra}")
