#!/usr/bin/env python3
"""Dump every action's kind in a target envelope — find Power index."""
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
    for j, a in enumerate(actions):
        if isinstance(a, dict):
            kind = a.get("action")
            print(f"{j:>3} {kind}")
            if kind == "Power":
                print(json.dumps(a, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
