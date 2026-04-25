"""Inspect the env 21 day 11 boundary in 1617442 — does the HP sync fire?"""
from __future__ import annotations
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    parse_p_envelopes_from_zip,
    map_snapshot_player_ids_to_engine,
)


def main() -> int:
    gid = 1617442
    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], 30, 12)
    print(f"awbw_to_engine: {awbw_to_engine}")

    # Failure says "P1 funds engine=3400 php_snapshot=3300 ... at (1, 14, 12) hp_bars engine=8 (hp=80) php_bars=9"
    # Tile diff key format = (seat, row, col). So row=14, col=12.
    seat_target = 1
    row_target, col_target = 14, 12

    for env_i in range(min(25, len(envs))):
        if (env_i + 1) >= len(frames):
            continue
        post_frame = frames[env_i + 1]
        units = post_frame.get("units") or {}
        if isinstance(units, dict):
            us = list(units.values())
        else:
            us = units
        for u in us:
            try:
                ux = int(u["x"]); uy = int(u["y"])
                upid = int(u["players_id"])
            except Exception:
                continue
            seat = awbw_to_engine.get(upid)
            if seat != seat_target:
                continue
            # PHP x=col y=row. Match if (ux,uy) == (col_target, row_target)?
            if ux == col_target and uy == row_target:
                hp_raw = u.get("hit_points")
                print(f"  env_i+1={env_i+1} day={post_frame.get('day')} PHP unit at (col={ux}, row={uy}) seat={seat} id={u.get('id')} hp={hp_raw}")
            # Also check the "diff key" interpretation: at (1, 14, 12) might mean (seat, row=14, col=12) OR (seat, x=14, y=12)
            if ux == row_target and uy == col_target:
                hp_raw = u.get("hit_points")
                print(f"  env_i+1={env_i+1} day={post_frame.get('day')} PHP unit at (col={ux}, row={uy}) [SWAPPED match] seat={seat} id={u.get('id')} hp={hp_raw}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
