"""Dump captures and tile context for env 17 of gid 1607045."""
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


def main() -> int:
    gid = 1607045
    co0, co1 = 5, 28
    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)

    target_env = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    pid, day, actions = envs[target_env]
    print(f"env {target_env} day {day} pid={pid}")
    for j, obj in enumerate(actions):
        kind = obj.get("action") or "?"
        if kind in ("Capt", "Fire", "AttackSeam", "End"):
            print(f"  [{j}] {kind}: {json.dumps({k: v for k, v in obj.items() if k not in ('combatInfoVision','discovered','combatInfo')}, default=str)[:300]}")
            if kind == "End":
                ui = obj.get("updatedInfo") or {}
                if isinstance(ui, dict):
                    for k, v in ui.items():
                        print(f"      End.updatedInfo[{k}] = {json.dumps(v, default=str)[:400]}")

    # also show next frame players_funds and properties around (0,11) and (6,16)
    f_after = frames[target_env + 1]
    props = f_after.get("properties") or f_after.get("buildings") or []
    print("\n-- properties near (0,11) and (6,16) in post-env frame --")
    if isinstance(props, dict):
        props = list(props.values())
    for p in props:
        try:
            x = int(p.get("x") or p.get("buildings_x") or 0)
            y = int(p.get("y") or p.get("buildings_y") or 0)
        except Exception:
            continue
        if (y, x) in {(0, 11), (6, 16)}:
            print(f"  pos=({y},{x}) -> {p}")

    # dump capt detail for current frame
    f_pre = frames[target_env]
    props_pre = f_pre.get("properties") or f_pre.get("buildings") or []
    if isinstance(props_pre, dict):
        props_pre = list(props_pre.values())
    print("\n-- properties near (0,11) and (6,16) in pre-env frame --")
    for p in props_pre:
        try:
            x = int(p.get("x") or p.get("buildings_x") or 0)
            y = int(p.get("y") or p.get("buildings_y") or 0)
        except Exception:
            continue
        if (y, x) in {(0, 11), (6, 16)}:
            print(f"  pos=({y},{x}) -> {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
