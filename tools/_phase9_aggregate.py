"""Phase 9 Lane L validation aggregate."""
import json
from collections import Counter

print('=== Phase 9 Lane L Validation Aggregate ===')
all_results = []
for n in (1, 2, 3, 4):
    path = f'logs/phase9_lane_l_val{n}_results.jsonl'
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    counts = Counter(r['verdict'] for r in rows)
    print(f'-- L-VAL-{n} ({len(rows)} rows): {dict(counts)}')
    all_results.extend(rows)

total_counts = Counter(r['verdict'] for r in all_results)
print()
print(f'=== TOTAL ({len(all_results)} rows) ===')
for k, v in total_counts.most_common():
    print(f'  {k}: {v}')

print()
print('=== ESCALATED_TO_ENGINE_BUG rows ===')
esc = [r for r in all_results if r['verdict'] == 'ESCALATED_TO_ENGINE_BUG']
for r in esc:
    gid = r['games_id']
    kind = r.get('new_action_kind', '?')
    msg = (r.get('new_message', '') or '')[:130]
    print(f'  gid={gid} kind={kind} msg={msg}')

print()
print('=== STUCK_SAME_FAMILY rows ===')
stuck = [r for r in all_results if r['verdict'] == 'STUCK_SAME_FAMILY']
print(f'  count: {len(stuck)}')
for r in stuck[:8]:
    gid = r['games_id']
    idx = r.get('new_envelope_index')
    msg = (r.get('new_message', '') or '')[:120]
    print(f'  gid={gid} env_idx={idx} msg={msg}')

print()
print('=== PROGRESSED_NEW_GAP message families (top 8) ===')
prog = [r for r in all_results if r['verdict'] == 'PROGRESSED_NEW_GAP']
def msg_family(m):
    m = m or ''
    if not m: return '<empty>'
    parts = m.split(':')
    return ':'.join(parts[:2])[:80]
fam = Counter(msg_family(r.get('new_message','')) for r in prog)
for k, v in fam.most_common(8):
    print(f'  [{v:>3}] {k}')

print()
print('=== CRASH or other ===')
other = [r for r in all_results if r['verdict'] not in ('FLIPPED_OK','PROGRESSED_NEW_GAP','STUCK_SAME_FAMILY','ESCALATED_TO_ENGINE_BUG')]
print(f'  count: {len(other)}')
for r in other[:5]:
    print(f'  gid={r["games_id"]} verdict={r["verdict"]} {(r.get("error","") or r.get("new_message","") or "")[:120]}')
