"""Dump every action in env 4 (P0 day 3) showing type, unit start, and final position."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.desync_audit import parse_p_envelopes_from_zip

zp = Path('replays/amarriner_gl/1631288.zip')
envelopes = parse_p_envelopes_from_zip(zp)
env4 = envelopes[4]
pid, day, actions = env4
print(f"env 4 pid={pid} day={day} #actions={len(actions)}")
for ai, a in enumerate(actions):
    if not isinstance(a, dict):
        print(f"  action {ai}: NON-DICT {type(a).__name__}: {str(a)[:200]}")
        continue
    kind = a.get('action')
    move = a.get('Move') or {}
    move_unit = None
    move_dest = None
    move_paths = None
    if isinstance(move, dict):
        mu = move.get('unit')
        if isinstance(mu, dict):
            for v in mu.values():
                if isinstance(v, dict):
                    for v2 in v.values():
                        if isinstance(v2, dict) and 'units_x' in v2:
                            move_unit = (v2.get('units_x'), v2.get('units_y'), v2.get('units_id'))
                            break
                    if move_unit is None and 'units_x' in v:
                        move_unit = (v.get('units_x'), v.get('units_y'), v.get('units_id'))
                if move_unit:
                    break
        move_paths = move.get('paths') or move.get('path')
        if isinstance(move_paths, list) and move_paths:
            last = move_paths[-1]
            if isinstance(last, dict):
                move_dest = (last.get('x'), last.get('y'))
    # extract Capt target if present
    capt = a.get('Capt') or {}
    capt_target = None
    if isinstance(capt, dict):
        bi = capt.get('buildingInfo') or {}
        if isinstance(bi, dict):
            capt_target = (bi.get('buildings_x'), bi.get('buildings_y'))
    print(f"  action {ai}: kind={kind} unit={move_unit} move_dest={move_dest} capt_target={capt_target}")
    # If unit ends near (17,4), dump full action
    if move_dest == (17, 4) or capt_target == (17, 4):
        print(f"    >>> FULL ACTION:")
        print("    " + json.dumps(a, default=str)[:2000])
