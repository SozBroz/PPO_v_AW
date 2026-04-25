#!/usr/bin/env python3
"""Drill Rachel SCOP unitReplace to confirm damage shape (3HP per missile, 3x3?)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip
ZIPS = ROOT / "replays" / "amarriner_gl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env", type=int, required=True)
    args = ap.parse_args()
    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    pid, day, actions = envs[args.env]
    pre_pid, pre_day, pre_actions = envs[args.env - 1] if args.env > 0 else (None, None, [])
    # Print Power and unitReplace
    for j, a in enumerate(actions):
        if isinstance(a, dict) and a.get("action") == "Power":
            print(f"=== env={args.env} idx={j} pid={pid} day={day} ===")
            print(f"missileCoords: {json.dumps(a.get('missileCoords'), indent=2)}")
            ur = a.get("unitReplace")
            if ur:
                print(f"unitReplace keys: {list(ur.keys()) if isinstance(ur, dict) else type(ur)}")
                print(json.dumps(ur, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
