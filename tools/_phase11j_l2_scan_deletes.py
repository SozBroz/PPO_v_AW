#!/usr/bin/env python3
"""Scan a zip for every Delete action; report whether same envelope has a Build."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

gid = int(sys.argv[1])
zpath = ROOT / "replays" / "amarriner_gl" / f"{gid}.zip"
envs = parse_p_envelopes_from_zip(zpath)
for env_i, (pid, day, actions) in enumerate(envs):
    kinds = [a.get("action") if isinstance(a, dict) else None for a in actions]
    if "Delete" not in kinds:
        continue
    has_build = "Build" in kinds
    # positions of Deletes & Builds
    positions = [(i, k) for i, k in enumerate(kinds) if k in ("Delete", "Build")]
    print(f"env={env_i} day={day} pid={pid} has_build={has_build} positions={positions}")
