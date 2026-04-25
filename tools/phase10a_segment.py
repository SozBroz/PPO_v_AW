"""Phase 10A: segment engine_bug rows by unit type and drift."""
import json
import re
from collections import Counter

rows = []
with open("logs/desync_register_post_phase9.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("class") != "engine_bug":
            continue
        msg = r.get("message", "") or ""
        m = re.search(
            r"for (\w+) from \((\d+), (\d+)\) \(unit_pos=\((\d+), (\d+)\)\)",
            msg,
        )
        if m:
            unit_type = m.group(1)
            from_pos = (int(m.group(2)), int(m.group(3)))
            unit_pos = (int(m.group(4)), int(m.group(5)))
            r["_drift"] = abs(from_pos[0] - unit_pos[0]) + abs(from_pos[1] - unit_pos[1])
            r["_from"] = from_pos
            r["_unit_pos"] = unit_pos
            r["_unit_type"] = unit_type
            rows.append(r)

by_unit = Counter(r["_unit_type"] for r in rows)
print("Unit distribution:", by_unit)
print()
b_copter_rows = [r for r in rows if r["_unit_type"] == "B_COPTER"]
b_copter_rows.sort(key=lambda r: (r["_drift"], int(r["games_id"])))
print(f"B_COPTER rows: {len(b_copter_rows)}")
print("drift distribution:", Counter(r["_drift"] for r in b_copter_rows))
print()
print("First 15 (smallest drift):")
for r in b_copter_rows[:15]:
    print(
        f"  gid={r['games_id']} drift={r['_drift']} from={r['_from']} "
        f"unit_pos={r['_unit_pos']} action_idx={r.get('action_index')}"
    )

with open("logs/phase10a_b_copter_targets.jsonl", "w", encoding="utf-8") as f:
    for r in b_copter_rows:
        f.write(json.dumps(r) + "\n")
print()
print("Wrote logs/phase10a_b_copter_targets.jsonl")

# Also write per-other-unit smallest drift rows for Step 6
print()
print("Other unit smallest-drift case study candidates:")
for ut in ("MECH", "RECON", "MEGA_TANK", "BLACK_BOAT"):
    subset = [r for r in rows if r["_unit_type"] == ut]
    subset.sort(key=lambda r: (r["_drift"], int(r["games_id"])))
    if subset:
        r = subset[0]
        print(
            f"  {ut}: gid={r['games_id']} drift={r['_drift']} "
            f"from={r['_from']} unit_pos={r['_unit_pos']} "
            f"action_idx={r.get('action_index')} (n={len(subset)})"
        )

with open("logs/phase10a_other_unit_targets.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        if r["_unit_type"] != "B_COPTER":
            f.write(json.dumps(r) + "\n")
print()
print("Wrote logs/phase10a_other_unit_targets.jsonl")
