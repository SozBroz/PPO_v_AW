#!/usr/bin/env python3
"""Phase 11K-CAPTURE-ACTIONS — dump every action whose payload touches
position (row,col) for gid 1635679 — to find why Sturm's capture of
(7,9) never completes engine-side."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

ZIPS = ROOT / "replays" / "amarriner_gl"


def _touches(obj, target):
    """Return True if any 'global' / x,y in obj or its sub-dicts targets pos."""
    if isinstance(obj, dict):
        x = obj.get('x'); y = obj.get('y')
        try:
            if (int(y), int(x)) == target:
                return True
        except (TypeError, ValueError):
            pass
        for v in obj.values():
            if _touches(v, target):
                return True
    elif isinstance(obj, list):
        for v in obj:
            if _touches(v, target):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, default=1635679)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--col", type=int, required=True)
    args = ap.parse_args()
    target = (args.row, args.col)

    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)

    for env_i, (pid, day, actions) in enumerate(envs):
        for ai, obj in enumerate(actions):
            if not _touches(obj, target):
                continue
            kind = obj.get('action') if isinstance(obj, dict) else '?'
            print(f"=== env={env_i} day={day} pid={pid} ai={ai} kind={kind} ===")
            print(json.dumps(obj, indent=2, default=str)[:2000])
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
