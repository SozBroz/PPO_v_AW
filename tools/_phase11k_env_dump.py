#!/usr/bin/env python3
"""Dump all actions in a single envelope concisely."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip

ZIPS = ROOT / "replays" / "amarriner_gl"


def _summarize(obj):
    if not isinstance(obj, dict):
        return repr(obj)[:60]
    kind = obj.get("action") or "?"
    out = f"action={kind}"
    if kind in ("Move",):
        unit = (obj.get("unit") or {}).get("global") or {}
        out += f" id={unit.get('units_id')} name={unit.get('units_name')} from=({unit.get('units_y')},{unit.get('units_x')})"
        path = obj.get("paths") or obj.get("path") or []
        if path:
            try:
                end = path[-1]
                out += f" -> ({end.get('y')},{end.get('x')})"
            except Exception:
                pass
        return out
    if kind == "Capt":
        bi = obj.get("buildingInfo") or {}
        out += f" pos=({bi.get('buildings_y')},{bi.get('buildings_x')}) cap_left={bi.get('buildings_capture')}"
        return out
    if kind in ("Fire", "AttackSeam"):
        ci = obj.get("combatInfo") or {}
        a = ci.get("attacker") or {}
        d = ci.get("defender") or {}
        out += f" att=({a.get('y')},{a.get('x')}) id={a.get('units_id')} hp_after={a.get('units_hit_points')}"
        out += f"  def=({d.get('y')},{d.get('x')}) id={d.get('units_id')} hp_after={d.get('units_hit_points')}"
        return out
    if kind == "Build":
        u = obj.get("unit") or {}
        out += f" pos=({u.get('y')},{u.get('x')}) name={u.get('units_name')}"
        return out
    if kind == "End":
        out += " (end-turn)"
        return out
    if kind == "Power":
        out += f" co={obj.get('coName')} cop={obj.get('isCopActive')}"
        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, default=1635679)
    ap.add_argument("--env", type=int, required=True)
    args = ap.parse_args()

    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    pid, day, actions = envs[args.env]
    print(f"env={args.env} pid={pid} day={day} #actions={len(actions)}")
    for ai, obj in enumerate(actions):
        print(f"  ai={ai}: {_summarize(obj)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
