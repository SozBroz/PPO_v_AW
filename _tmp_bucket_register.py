import json
from collections import Counter

buckets = Counter()
samples = {}
with open('logs/desync_register.jsonl', encoding='utf-8') as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('class') not in ('oracle_gap', 'engine_bug'):
            continue
        msg = r.get('message') or ''
        if 'Fire (no path)' in msg or 'Fire: no attacker' in msg or 'AttackSeam' in msg:
            b = 'fire_no_path'
        elif 'Move: no unit' in msg:
            b = 'move_no_unit'
        elif 'Capt' in msg:
            b = 'capt_drift'
        elif 'Unload' in msg:
            b = 'unload'
        elif 'Repair' in msg:
            b = 'repair'
        elif 'Build' in msg:
            b = 'build'
        elif 'Power' in msg or 'COP' in msg or 'SCOP' in msg:
            b = 'power'
        elif 'Malformed' in msg or 'invalid literal' in msg:
            b = 'malformed_json'
        else:
            b = 'other'
        buckets[b] += 1
        samples.setdefault(b, []).append((r.get('games_id'), msg[:160]))
for k, v in buckets.most_common():
    print(f'{k:20} {v}')
print()
for k, exs in samples.items():
    print(f'--- {k} ({len(exs)}) ---')
    for gid, m in exs:
        print(f'  {gid}: {m}')
