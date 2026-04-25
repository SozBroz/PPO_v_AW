#!/usr/bin/env python3
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / 'logs/phase11j_envinspect2.json', encoding='utf-8-sig'))
for c in d:
    print('====', c['gid'], 'env', c['env_idx'], 'j', c['fail_j'])
    fire = c['Fire']
    move = c['Move']
    civ = fire.get('combatInfoVision', {})
    glob = civ.get('global', {})
    ci = glob.get('combatInfo', {})
    att = ci.get('attacker', {})
    de = ci.get('defender', {})
    gu = (move.get('unit') or {}).get('global') if isinstance(move.get('unit'), dict) else None
    paths = (move.get('paths') or {}).get('global') or []
    if gu:
        print(f"  Move.unit.global: id={gu.get('units_id')} type={gu.get('units_name')} (y,x)=({gu.get('units_y')},{gu.get('units_x')}) ammo={gu.get('units_ammo')} fuel={gu.get('units_fuel')}")
    else:
        print('  Move.unit.global: NONE (no-path)')
    if paths:
        print(f"  paths n={len(paths)} start=({paths[0]['y']},{paths[0]['x']}) end=({paths[-1]['y']},{paths[-1]['x']})")
    else:
        print('  paths: NONE')
    print(f"  attacker: id={att.get('units_id')} (y,x)=({att.get('units_y')},{att.get('units_x')}) ammo={att.get('units_ammo')} hp={att.get('units_hit_points')}")
    print(f"  defender: id={de.get('units_id')} (y,x)=({de.get('units_y')},{de.get('units_x')}) ammo={de.get('units_ammo')} hp={de.get('units_hit_points')}")
