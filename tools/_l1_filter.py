"""Quick filter to extract the 25 BUILD-FUNDS-RESIDUAL rows."""
import json, sys, collections

rows = []
with open('logs/desync_register_post_phase11j_v2_936.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

build = [r for r in rows if r.get('class')=='oracle_gap' and 'Build no-op' in r.get('message','') and 'insufficient funds' in r.get('message','')]
print(f"total rows: {len(rows)}")
print(f"BUILD-FUNDS-RESIDUAL count: {len(build)}")
print()
for i, r in enumerate(build):
    print(f"--- {i+1} ---")
    for k, v in r.items():
        print(f"  {k}: {v}")
    print()
