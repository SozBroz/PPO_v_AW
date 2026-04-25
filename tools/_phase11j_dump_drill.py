#!/usr/bin/env python3
"""Dump all per-envelope rows for a specific gid from the drill JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRILL = ROOT / "logs" / "phase11j_buildnoop12_drill.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--input", type=Path, default=DRILL)
    args = ap.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for c in data["cases"]:
        if c["gid"] != args.gid:
            continue
        for r in c.get("per_envelope") or []:
            if "fail_msg" in r:
                print(f"FAIL env={r.get('env_i')} day={r.get('day')} "
                      f"eng={r.get('engine_funds')} pre_php={r.get('php_funds_pre_env')} "
                      f"post_php={r.get('php_funds_post_env')}")
                print(f"  {r.get('fail_msg')}")
            else:
                print(f"env={r['env_i']:>3} day={r.get('day')} "
                      f"pid={r.get('pid')} delta={r.get('delta_engine_minus_php')} "
                      f"eng={r.get('engine_funds')} php={r.get('php_funds')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
