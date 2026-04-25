"""Dump full Fire action structure for env 17 of gid 1607045."""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    parse_p_envelopes_from_zip,
    map_snapshot_player_ids_to_engine,
)


TARGET_PHP_IDS = {190277871, 190289865}


def main() -> int:
    gid = 1607045
    co0, co1 = 5, 28
    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)

    pid, day, actions = envs[17]
    print(f"env 17 day {day} pid={pid}")

    for j, obj in enumerate(actions):
        if obj.get("action") != "Fire":
            continue
        keys = list(obj.keys())
        print(f"\n[{j}] Fire keys: {keys}")
        fire_blk = obj.get("Fire") or {}
        print(f"    Fire keys: {list(fire_blk.keys()) if isinstance(fire_blk, dict) else type(fire_blk)}")
        if isinstance(fire_blk, dict):
            obj = fire_blk  # keep going below using the inner Fire block
        # combatInfoVision is the primary location
        civ = obj.get("combatInfoVision") or {}
        if isinstance(civ, dict):
            for k, v in civ.items():
                if isinstance(v, dict):
                    inner_keys = list(v.keys())
                    print(f"    civ[{k}] keys: {inner_keys}")
                    inner = v.get("combatInfo") or v
                    a = (inner or {}).get("attacker") if isinstance(inner, dict) else None
                    d = (inner or {}).get("defender") if isinstance(inner, dict) else None
                    if isinstance(a, dict):
                        print(f"      attacker uid={a.get('units_id')} {a.get('units_name')} hp={a.get('units_hit_points')} pos=({a.get('units_y')},{a.get('units_x')})")
                    if isinstance(d, dict):
                        print(f"      defender uid={d.get('units_id')} {d.get('units_name')} hp={d.get('units_hit_points')} pos=({d.get('units_y')},{d.get('units_x')})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
