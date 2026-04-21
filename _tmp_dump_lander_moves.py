"""Dump all Lander 192407337 Move paths."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip

ZIP = Path('replays/amarriner_gl/1631302.zip')
LANDER_ID = 192407337
envelopes = parse_p_envelopes_from_zip(ZIP)

for ei, (pid, day, actions) in enumerate(envelopes):
    for ai, act in enumerate(actions):
        if act.get('action') != 'Move':
            continue
        unit = act.get('unit') or {}
        for k, v in unit.items():
            if isinstance(v, dict) and v.get('units_id') == LANDER_ID:
                paths = act.get('paths') or {}
                p_glob = (paths.get('global') if isinstance(paths, dict) else None) or []
                fuel = v.get('units_fuel')
                print(f'env={ei} day={day} ai={ai}: fuel={fuel} path={[(p.get("x"),p.get("y")) for p in p_glob]}')
                break
