#!/usr/bin/env python3
"""Dump RAW env 24 + env 25 Capt actions with full JSON for gid 1628849."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

GID = 1628849
zpath = ROOT / "replays" / "amarriner_gl" / f"{GID}.zip"
envs = parse_p_envelopes_from_zip(zpath)
for env_i in (24, 25):
    pid, day, actions = envs[env_i]
    print(f"\n========== ENV {env_i} pid={pid} day={day} n={len(actions)} ==========")
    for j, obj in enumerate(actions):
        kind = obj.get("action") or obj.get("type")
        if kind in ("Capt", "Build", "Power"):
            print(f"  --- [{j}] {kind} ---")
            print(json.dumps(obj, indent=2, default=str)[:1500])
