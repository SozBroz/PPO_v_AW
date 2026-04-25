"""Dump all Fire-action attackers/defenders in a target envelope."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.oracle_zip_replay import parse_p_envelopes_from_zip


def main() -> None:
    gid = int(sys.argv[1])
    target_env = int(sys.argv[2])
    envs = parse_p_envelopes_from_zip(Path(f"replays/amarriner_gl/{gid}.zip"))
    pid, day, actions = envs[target_env]
    print(f"env={target_env} pid={pid} day={day} actions={len(actions)}")
    for ai, obj in enumerate(actions):
        kind = obj.get("action") or "?"
        if kind != "Fire":
            continue
        fire = obj.get("Fire") or {}
        civ = fire.get("combatInfoVision") or {}
        # Prefer global view; fall back to any view that has combatInfo.
        ci = None
        for k in ("global", str(pid), *list(civ.keys())):
            v = civ.get(k)
            if isinstance(v, dict) and v.get("combatInfo"):
                ci = v["combatInfo"]
                break
        atk = (ci or {}).get("attacker") or {}
        d = (ci or {}).get("defender") or {}
        # Move sub-action info
        move = obj.get("Move")
        m_uid = m_x = m_y = None
        if isinstance(move, dict):
            unit = move.get("unit") or {}
            glb = unit.get("global") if isinstance(unit, dict) else None
            if isinstance(glb, dict):
                m_uid = glb.get("units_id")
                m_x = glb.get("units_x")
                m_y = glb.get("units_y")
        print(
            f"  ai={ai:2d} ATK uid={atk.get('units_id')} ammo={atk.get('units_ammo')} hp={atk.get('units_hit_points')} xy=({atk.get('units_x')},{atk.get('units_y')})"
            f"  DEF uid={d.get('units_id')} hp={d.get('units_hit_points')} xy=({d.get('units_x')},{d.get('units_y')})"
            f"  Move(uid={m_uid} xy=({m_x},{m_y}))"
        )


if __name__ == "__main__":
    main()
