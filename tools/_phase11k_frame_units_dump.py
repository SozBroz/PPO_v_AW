#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.diff_replay_zips import load_replay

zpath = ROOT / "replays" / "amarriner_gl" / "1635679.zip"
frames = load_replay(zpath)
f = frames[11]
print(f"frame[11] keys: {list(f.keys())}")
units = f.get("units") or {}
print(f"units keys (first 5): {list(units.keys())[:5]}")
# Pick the recon at (3,4)
for k, u in units.items():
    if int(u["y"]) == 3 and int(u["x"]) == 4:
        print(f"recon at (3,4) — key={k!r}")
        print(json.dumps(u, indent=2, sort_keys=True, default=str)[:1500])
        break
