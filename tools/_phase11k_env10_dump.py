#!/usr/bin/env python3
"""Dump full action JSON for env 10 of gid 1635679."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

zpath = ROOT / "replays" / "amarriner_gl" / "1635679.zip"
envs = parse_p_envelopes_from_zip(zpath)
pid, day, actions = envs[10]
print(f"env=10 pid={pid} day={day} n_actions={len(actions)}\n")
for i, a in enumerate(actions):
    print(f"--- ai={i} ---")
    print(json.dumps(a, indent=2, sort_keys=True, default=str)[:2000])
    print()
