"""Quick desync status drilldown — engine_bug rows + oracle_gap families."""
import json
import collections
from pathlib import Path

LATEST = "logs/desync_register_l2_postfix_936.jsonl"

print(f"=== Engine bugs in {LATEST} ===")
with open(LATEST, encoding="utf-8") as f:
    for ln in f:
        r = json.loads(ln)
        if r.get("class") == "engine_bug":
            gid = r["games_id"]
            day = r.get("approx_day")
            kind = r.get("approx_action_kind")
            msg = (r.get("message") or "")[:140]
            cos = f'CO_p0={r.get("co_p0_id")} CO_p1={r.get("co_p1_id")}'
            print(f"  gid={gid} day={day} kind={kind} {cos}")
            print(f"    msg={msg}")

print()
print(f"=== Oracle gap message families in {LATEST} ===")
mc = collections.Counter()
with open(LATEST, encoding="utf-8") as f:
    for ln in f:
        r = json.loads(ln)
        if r.get("class") == "oracle_gap":
            msg = (r.get("message") or "").split(":", 1)[0][:90]
            mc[msg] += 1
for k, v in mc.most_common():
    print(f"  {v:3d}  {k}")

print()
print("=== State-mismatch retuned register engine_bug-equivalents ===")
SMI = "logs/desync_register_state_mismatch_936_retune.jsonl"
if Path(SMI).exists():
    cls_count = collections.Counter()
    sub_funds = collections.Counter()
    sub_units = collections.Counter()
    with open(SMI, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            cls = r.get("class") or "null"
            cls_count[cls] += 1
            if cls == "state_mismatch_funds":
                msg = (r.get("message") or "").split(":", 1)[0][:90]
                sub_funds[msg] += 1
            if cls == "state_mismatch_units":
                msg = (r.get("message") or "").split(":", 1)[0][:90]
                sub_units[msg] += 1
    print(f"  Class breakdown: {dict(cls_count)}")
    print(f"  Top funds families:")
    for k, v in sub_funds.most_common(5):
        print(f"    {v:3d}  {k}")
    print(f"  Top units families:")
    for k, v in sub_units.most_common(5):
        print(f"    {v:3d}  {k}")
