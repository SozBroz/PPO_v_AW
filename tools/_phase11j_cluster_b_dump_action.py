#!/usr/bin/env python3
"""Dump a specific Fire action JSON from a target envelope (by idx)."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env", type=int, required=True)
    ap.add_argument("--idx", type=int, required=True)
    args = ap.parse_args()

    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    pid, day, actions = envs[args.env]
    act = actions[args.idx]
    print(f"pid={pid} day={day} idx={args.idx} kind={act.get('action')}")
    print(json.dumps(act, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
