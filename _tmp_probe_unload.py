"""Track Lander 192407337 around env=40 day=21 — failed unload to (8,3)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip

ZIP = Path('replays/amarriner_gl/1631302.zip')
LANDER_ID = 192407337

envelopes = parse_p_envelopes_from_zip(ZIP)

# Print all actions in env 38, 39, 40 (full context)
for ei in [37, 38, 39, 40, 41]:
    if ei >= len(envelopes):
        continue
    pid, day, actions = envelopes[ei]
    print(f'\n=== env={ei} pid={pid} day={day} ({len(actions)} actions) ===')
    for ai, act in enumerate(actions):
        kind = act.get('action')
        text = json.dumps(act)
        marker = '*' if str(LANDER_ID) in text else ' '
        # Brief
        if kind == 'Move':
            unit = act.get('unit') or {}
            for k, v in unit.items():
                if isinstance(v, dict):
                    uid = v.get('units_id')
                    nm = v.get('units_name')
                    y = v.get('units_y')
                    x = v.get('units_x')
                    paths = act.get('paths') or act.get('path')
                    last = None
                    if paths:
                        if isinstance(paths, list) and paths:
                            last = paths[-1]
                        elif isinstance(paths, dict):
                            keys = sorted([int(k) for k in paths.keys() if str(k).isdigit()])
                            if keys:
                                last = paths.get(str(keys[-1])) or paths.get(keys[-1])
                    print(f'  {marker} ai={ai} Move {nm} id={uid} from (y={y},x={x}) path-end={last}')
                    break
        elif kind == 'Load':
            move = act.get('Move') or {}
            mu = move.get('unit') or {}
            for k, v in mu.items():
                if isinstance(v, dict):
                    uid = v.get('units_id')
                    nm = v.get('units_name')
                    y = v.get('units_y')
                    x = v.get('units_x')
                    print(f'  {marker} ai={ai} Load (cargo {nm} id={uid} from y={y},x={x})')
                    break
        elif kind == 'Unload':
            unit = act.get('unit') or {}
            cargo = unit.get('global') or {}
            print(f'  {marker} ai={ai} Unload {cargo.get("units_name")} to (y={cargo.get("units_y")},x={cargo.get("units_x")}) tid={act.get("transportID")}')
        elif kind in ('Build', 'Capt', 'Fire', 'End', 'Power'):
            print(f'  {marker} ai={ai} {kind}: {text[:140]}')
        else:
            print(f'  {marker} ai={ai} {kind}')
