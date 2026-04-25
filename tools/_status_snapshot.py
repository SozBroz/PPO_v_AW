"""One-off status snapshot of all desync registers."""
import json
import os
from collections import Counter

def stats(path, label):
    if not os.path.exists(path):
        print(f"{label:<55}  (missing)")
        return
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    cls_key = "class" if rows and "class" in rows[0] else "cls"
    counts = Counter(r.get(cls_key) for r in rows)
    total = len(rows)
    eb = counts.get("engine_bug", 0)
    og = counts.get("oracle_gap", 0)
    ok = counts.get("ok", 0)
    other = total - ok - og - eb
    extra = f"  other={other}" if other else ""
    print(f"{label:<55} total={total:>4}  ok={ok:>3}  oracle_gap={og:>3}  engine_bug={eb:>3}{extra}")
    if 0 < eb <= 15:
        for r in rows:
            if r.get(cls_key) == "engine_bug":
                msg = (r.get("message", "") or "")[:90]
                gid = r.get("games_id", "?")
                day = r.get("approx_day", "?")
                kind = r.get("approx_action_kind", "?")
                print(f"    eb gid={gid} day={day} kind={kind}: {msg}")

print("=== HISTORICAL BASELINES ===")
stats("logs/desync_register_post_phase10q.jsonl", "Phase 10Q (741 std, pre-Phase-11)")
stats("logs/desync_register_extras_baseline.jsonl", "Phase 11W (195 extras only)")
stats("logs/desync_register_combined_936.jsonl", "Phase 11Z combined 936 (post 11A/B/C, pre J)")
print()
print("=== POST-PHASE-11J (FIRE-DRIFT + KOAL landed) ===")
stats("logs/desync_register_post_phase11j_f2_100.jsonl", "KOAL only: first 100 (seed 1)")
stats("logs/desync_register_post_phase11j_sample.jsonl", "FIRE-DRIFT first 100 (after Edits A/B/C)")
stats("logs/desync_register_post_phase11j_fire_drift_50.jsonl", "FIRE-DRIFT-CLOSEOUT 50 sample (LATEST)")
stats("logs/desync_register_post_phase11j_fire_drift_targeted.jsonl", "FIRE-DRIFT 7 targets (LATEST)")
stats("logs/desync_register_post_phase11j_fire_drift_koal_crosscheck.jsonl", "KOAL targets cross-check (LATEST)")
stats("logs/desync_register_post_phase11j_combined.jsonl", "11J combined-so-far (PARTIAL)")
