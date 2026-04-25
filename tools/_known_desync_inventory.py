"""Quick inventory of remaining engine_bug + oracle_gap rows from the latest register."""
import json
import sys
from collections import defaultdict

REG = sys.argv[1] if len(sys.argv) > 1 else "logs/desync_register_FINAL_936_20260421_1422.jsonl"

rows = [json.loads(l) for l in open(REG, encoding="utf-8")]
gaps = [r for r in rows if r.get("class") == "oracle_gap"]
bugs = [r for r in rows if r.get("class") == "engine_bug"]

print(f"REGISTER: {REG}")
print(f"== ENGINE_BUG ({len(bugs)}) ==")
for r in bugs:
    msg = (r.get("message") or "")[:120]
    print(f"  gid={r['games_id']:>7} d{r.get('approx_day')} "
          f"{r.get('approx_action_kind'):<12} | {msg}")

print(f"\n== ORACLE_GAP ({len(gaps)}) ==")
fam = defaultdict(list)
for r in gaps:
    msg = (r.get("message") or "")[:70]
    fam[msg].append(r["games_id"])
for k, v in sorted(fam.items(), key=lambda x: -len(x[1])):
    print(f"  [{len(v)}] {k}")
    print(f"        gids: {v}")
