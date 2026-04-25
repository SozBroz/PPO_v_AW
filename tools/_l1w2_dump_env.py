#!/usr/bin/env python3
"""Dump a specific envelope's actions for triage."""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env", type=int, required=True)
    args = ap.parse_args()
    envs = parse_p_envelopes_from_zip(ROOT / "replays" / "amarriner_gl" / f"{args.gid}.zip")
    pid, day, actions = envs[args.env]
    print(f"--- env {args.env} pid={pid} day={day} actions={len(actions)} ---")
    for j, a in enumerate(actions):
        kind = a.get("action") or a.get("type")
        info = ""
        if kind == "Build":
            gu = (a.get("Build", {}).get("newUnit", {}) or {}).get("global", {})
            info = f"{gu.get('units_name')} at ({gu.get('units_y')},{gu.get('units_x')}) units_id={gu.get('units_id')} cost_field={gu.get('units_cost', '?')}"
        elif kind == "Capt":
            bi = (a.get("Capt", {}).get("buildingInfo") or {})
            info = f"tile=({bi.get('buildings_y')},{bi.get('buildings_x')}) cap={bi.get('buildings_capture')}"
        elif kind == "Fire":
            ci = (a.get("Fire", {}).get("combatInfoVision", {}) or {}).get("global", {}).get("combatInfo", {})
            atk = ci.get("attacker", {})
            df = ci.get("defender", {})
            info = (
                f"atk={atk.get('units_name')} hp{atk.get('units_hit_points')}->? "
                f"def={df.get('units_name')} hp{df.get('units_hit_points')}->?"
            )
        elif kind == "Power":
            p = a.get("Power", {})
            info = f"co={p.get('coName')} kind={p.get('coPower')}"
        elif kind == "End":
            info = f"nextFunds={a.get('End', {}).get('updatedInfo', {}).get('nextFunds', {}).get('global')}"
        elif kind == "Move":
            u = (a.get("unit", {}) or {})
            for v in u.values():
                if isinstance(v, dict):
                    info = f"{v.get('units_name')} fuel={v.get('units_fuel')}"
                    break
        print(f"  [{j:2}] {kind:10s} {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
