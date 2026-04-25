"""Classify the 25 BUILD-FUNDS-RESIDUAL rows by active CO + opponent CO + power state.

Columns:
  gid, day, active_seat, active_co, active_co_name, opp_co, opp_co_name,
  unit, need, have, shortfall
"""
import json
import re
from collections import Counter, defaultdict

with open('data/co_data.json', encoding='utf-8') as f:
    co_data = json.load(f)['cos']
co_name = {int(k): v['name'] for k, v in co_data.items()}

rows = []
with open('logs/desync_register_post_phase11j_v2_936.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

build = [r for r in rows if r.get('class')=='oracle_gap' and 'Build no-op' in r.get('message','') and 'insufficient funds' in r.get('message','')]

pat = re.compile(r"unit=(\w+) for engine (P\d+):.*need (\d+)\$, have (\d+)\$")

print(f"{'#':>2} {'gid':>8} {'day':>3} {'seat':>4} {'a_co':>4} {'a_name':>10} {'o_co':>4} {'o_name':>10} {'unit':>10} {'need':>6} {'have':>6} {'short':>6}")
print('-' * 110)
records = []
for i, r in enumerate(build):
    m = pat.search(r['message'])
    if not m:
        print('NO MATCH:', r['message']); continue
    unit, seat, need, have = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    seat_idx = int(seat[1:])
    a_co = r['co_p0_id'] if seat_idx == 0 else r['co_p1_id']
    o_co = r['co_p1_id'] if seat_idx == 0 else r['co_p0_id']
    rec = dict(idx=i+1, gid=r['games_id'], day=r['approx_day'], seat=seat,
               a_co=a_co, a_name=co_name.get(a_co,'?'),
               o_co=o_co, o_name=co_name.get(o_co,'?'),
               unit=unit, need=need, have=have, short=need-have,
               zip=r['zip_path'], env=r['approx_envelope_index'])
    records.append(rec)
    print(f"{rec['idx']:>2} {rec['gid']:>8} {rec['day']:>3} {seat:>4} {a_co:>4} {rec['a_name']:>10} {o_co:>4} {rec['o_name']:>10} {unit:>10} {need:>6} {have:>6} {rec['short']:>6}")

print()
print('=== ACTIVE CO TALLY ===')
ctr = Counter((r['a_co'], r['a_name']) for r in records)
for (co, name), n in ctr.most_common():
    print(f"  {co:>3} {name:>12}: {n}")

print()
print('=== OPPONENT CO TALLY ===')
ctr = Counter((r['o_co'], r['o_name']) for r in records)
for (co, name), n in ctr.most_common():
    print(f"  {co:>3} {name:>12}: {n}")

print()
print('=== PAIR (active, opp) TALLY ===')
ctr = Counter((r['a_name'], r['o_name']) for r in records)
for k, n in ctr.most_common():
    print(f"  {k}: {n}")

print()
print('=== UNIT TALLY ===')
print(Counter(r['unit'] for r in records).most_common())

print()
print('=== SHORTFALL DISTRIBUTION ===')
shorts = sorted(r['short'] for r in records)
print(f"  min={shorts[0]} median={shorts[len(shorts)//2]} max={shorts[-1]} mean={sum(shorts)/len(shorts):.0f}")
print(f"  shortfalls: {shorts}")

# Save records for next step
with open('logs/_l1_records.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2)
print()
print(f"Saved {len(records)} records to logs/_l1_records.json")
