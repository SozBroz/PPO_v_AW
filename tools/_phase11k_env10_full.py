#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

zpath = ROOT / "replays" / "amarriner_gl" / "1635679.zip"
envs = parse_p_envelopes_from_zip(zpath)
pid, day, actions = envs[10]
a = actions[0]
print(json.dumps(a, indent=2, sort_keys=True, default=str))
