"""Drilldown of the post-wave5 936 register."""
import json
import collections
import sys
from pathlib import Path

NEW = "logs/desync_register_post_wave5_936_20260421_1335.jsonl"
PREV = "logs/desync_register_l2_postfix_936.jsonl"

def load(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows

new = load(NEW)
prev = load(PREV)

def by_gid(rows):
    return {r["games_id"]: r for r in rows}

n = by_gid(new)
p = by_gid(prev)

print(f"=== TOTALS ===")
print(f"NEW  ({Path(NEW).name}): {len(new)} rows")
nc = collections.Counter(r.get("class") or "null" for r in new)
for k, v in nc.most_common():
    print(f"   {k}: {v}")
print(f"PREV ({Path(PREV).name}): {len(prev)} rows")
pc = collections.Counter(r.get("class") or "null" for r in prev)
for k, v in pc.most_common():
    print(f"   {k}: {v}")

print()
print("=== DELTA ===")
print(f"   ok:         {nc['ok']:4d}  ({nc['ok'] - pc['ok']:+d})")
print(f"   oracle_gap: {nc['oracle_gap']:4d}  ({nc['oracle_gap'] - pc['oracle_gap']:+d})")
print(f"   engine_bug: {nc['engine_bug']:4d}  ({nc['engine_bug'] - pc['engine_bug']:+d})")

print()
print("=== ENGINE BUGS in NEW ===")
for gid, r in n.items():
    if r.get("class") == "engine_bug":
        cos = f"CO_p0={r.get('co_p0_id')} CO_p1={r.get('co_p1_id')}"
        msg = (r.get("message") or "")[:160]
        print(f"  gid={gid} day={r.get('approx_day')} kind={r.get('approx_action_kind')} {cos}")
        print(f"    msg={msg}")

print()
print("=== ORACLE_GAP families in NEW ===")
mc = collections.Counter()
for r in new:
    if r.get("class") == "oracle_gap":
        msg = (r.get("message") or "").split(":", 1)[0][:90]
        mc[msg] += 1
for k, v in mc.most_common():
    print(f"  {v:3d}  {k}")

print()
print("=== CLASS FLIPS (NEW vs PREV) ===")
flips = collections.Counter()
flip_examples = collections.defaultdict(list)
for gid, nr in n.items():
    if gid not in p:
        continue
    pr = p[gid]
    nc_, pc_ = nr.get("class") or "?", pr.get("class") or "?"
    if nc_ != pc_:
        key = f"{pc_} -> {nc_}"
        flips[key] += 1
        if len(flip_examples[key]) < 3:
            flip_examples[key].append(gid)
for k, v in flips.most_common():
    print(f"  {v:3d}  {k}   examples: {flip_examples[k]}")
