#!/usr/bin/env python3
"""
Phase 11J — dump the failing Fire envelope's combatInfo + Move.global for each gid.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip  # noqa: E402

TARGETS = {
    1622104: 43,
    1625784: 35,
    1630983: 24,
    1631494: 46,
    1634664: 23,
    1635025: 36,
    1635846: 31,
}

def main() -> int:
    out = {}
    for gid, env_idx in TARGETS.items():
        zpath = ROOT / "replays" / "amarriner_gl" / f"{gid}.zip"
        envs = parse_p_envelopes_from_zip(zpath)
        if env_idx >= len(envs):
            out[gid] = {"error": f"env_idx {env_idx} >= {len(envs)}"}
            continue
        pid, day, actions = envs[env_idx]
        # find the failing Fire action — typically the last Fire action in the envelope
        # but we want to see all of them
        fires = []
        for j, obj in enumerate(actions):
            if obj.get("action") == "Fire":
                f = obj.get("Fire") or {}
                m = obj.get("Move") or {}
                ci = (f.get("combatInfoVision") or {}).get("global") or {}
                ci_combat = ci.get("combatInfo") or {}
                attacker = ci_combat.get("attacker") or {}
                defender = ci_combat.get("defender") or {}
                gu = (m.get("unit") or {}).get("global") if isinstance(m.get("unit"), dict) else {}
                paths = (m.get("paths") or {}).get("global") or []
                fires.append({
                    "j": j,
                    "attacker": {
                        "id": attacker.get("units_id"),
                        "type": attacker.get("units_name"),
                        "y": attacker.get("units_y"),
                        "x": attacker.get("units_x"),
                        "hp": attacker.get("units_hit_points"),
                        "ammo": attacker.get("units_ammo"),
                        "pid": attacker.get("units_players_id"),
                    },
                    "defender": {
                        "id": defender.get("units_id"),
                        "type": defender.get("units_name"),
                        "y": defender.get("units_y"),
                        "x": defender.get("units_x"),
                        "hp": defender.get("units_hit_points"),
                        "pid": defender.get("units_players_id"),
                    },
                    "move_global_unit": {
                        "id": (gu or {}).get("units_id"),
                        "type": (gu or {}).get("units_name"),
                        "y": (gu or {}).get("units_y"),
                        "x": (gu or {}).get("units_x"),
                        "ammo": (gu or {}).get("units_ammo"),
                        "pid": (gu or {}).get("units_players_id"),
                    } if gu else None,
                    "path_start": [paths[0].get("y"), paths[0].get("x")] if paths else None,
                    "path_end": [paths[-1].get("y"), paths[-1].get("x")] if paths else None,
                    "n_path": len(paths),
                })
        out[gid] = {
            "env_idx": env_idx,
            "envelope_pid": pid,
            "envelope_day": day,
            "envelope_action_kinds": [a.get("action") for a in actions],
            "fires": fires,
        }
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
