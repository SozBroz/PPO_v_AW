#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip
GID = 1628849
zpath = ROOT / "replays" / "amarriner_gl" / f"{GID}.zip"
envs = parse_p_envelopes_from_zip(zpath)
pid, day, actions = envs[25]
for j in (1, 2, 3, 4, 5):
    print(f"--- Action [{j}] full JSON ---")
    print(json.dumps(actions[j], indent=2, default=str)[:2000])
    print()
