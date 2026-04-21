"""Detailed trace of Lander 192407337 path at env=38 ai=21 and Load/Unload at env=40."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip

ZIP = Path('replays/amarriner_gl/1631302.zip')
LANDER_ID = 192407337

envelopes = parse_p_envelopes_from_zip(ZIP)

print("=== env=38 ai=21 (Lander Move) ===")
pid, day, actions = envelopes[38]
act = actions[21]
print(json.dumps(act, indent=2)[:3000])

print("\n=== env=40 ai=9 (Load Mech) ===")
pid, day, actions = envelopes[40]
act = actions[9]
print(json.dumps(act, indent=2)[:3000])

print("\n=== env=40 ai=10 (Unload Mech) ===")
act = actions[10]
print(json.dumps(act, indent=2)[:3000])
