#!/usr/bin/env python3
"""Dump the raw JSON of a specific (env, ai) action."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip
ZIPS = ROOT / "replays" / "amarriner_gl"

ap = argparse.ArgumentParser()
ap.add_argument("--gid", type=int, default=1635679)
ap.add_argument("--env", type=int, required=True)
ap.add_argument("--ai", type=int, required=True)
args = ap.parse_args()

envs = parse_p_envelopes_from_zip(ZIPS / f"{args.gid}.zip")
pid, day, actions = envs[args.env]
print(json.dumps(actions[args.ai], indent=2, default=str))
