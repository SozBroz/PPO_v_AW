"""Classify state_mismatch_funds rows for closeout doc."""
import json
from collections import Counter

c = Counter()
fund_rows = []
for line in open('logs/_lastmile_v2_state_mismatch_2.jsonl', encoding='utf-8'):
    r = json.loads(line)
    c[r.get('class')] += 1
    if r.get('class') == 'state_mismatch_funds':
        fund_rows.append(r)
print('summary:', dict(c))
print('---')
for r in sorted(
    fund_rows,
    key=lambda x: abs(x['state_mismatch']['diff_summary']['funds_delta_by_seat'].get('0', 0))
    + abs(x['state_mismatch']['diff_summary']['funds_delta_by_seat'].get('1', 0)),
    reverse=True,
):
    d = r['state_mismatch']['diff_summary']['funds_delta_by_seat']
    abs_d = abs(d.get('0', 0)) + abs(d.get('1', 0))
    has_hp = 'hp_bars' in r['message']
    print(
        f"  gid={r['games_id']} delta={abs_d:>5} day={r['approx_day']:>3} env={r['approx_envelope_index']:>3} "
        f"cop0={r['co_p0_id']:>3} cop1={r['co_p1_id']:>3} hp_drift={has_hp} msg={r['message'][:120]}"
    )
