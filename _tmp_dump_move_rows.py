import json
rows = []
with open('logs/desync_register.jsonl', encoding='utf-8') as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if 'Move: no unit' in (r.get('message') or ''):
            rows.append(r)
for r in rows:
    print(f"{r.get('games_id')}  day~{r.get('approx_day','?')} P0={r.get('co_p0_id','?')} P1={r.get('co_p1_id','?')}: {(r.get('message') or '')[:160]}")
print()
print('total:', len(rows))
