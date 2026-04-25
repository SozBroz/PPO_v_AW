#!/usr/bin/env python3
"""Phase 11J-RACHEL-SCOP — locate Rachel SCOP Power envelope and dump missileCoords."""
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
    args = ap.parse_args()
    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    found = 0
    for env_idx, (pid, day, actions) in enumerate(envs):
        for j, a in enumerate(actions):
            if isinstance(a, dict) and a.get("action") == "Power":
                co_name = a.get("coName")
                co_power = a.get("coPower")
                mc = a.get("missileCoords")
                if co_name == "Rachel" and co_power == "S":
                    print(f"=== env={env_idx} action_idx={j} pid={pid} day={day} ===")
                    print(f"coName={co_name} coPower={co_power}")
                    print(f"missileCoords={json.dumps(mc, indent=2)}")
                    print(f"all keys: {sorted(a.keys())}")
                    found += 1
    if not found:
        print("NO Rachel SCOP found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
