"""Classify state_mismatch_funds rows from a desync_audit register."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "_lastmile_v2_state_mismatch_1.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    funds = [r for r in rows if r.get("class") == "state_mismatch_funds"]
    multi = [r for r in rows if r.get("class") == "state_mismatch_multi"]
    units = [r for r in rows if r.get("class") == "state_mismatch_units"]
    print(f"funds_only_count={len(funds)}")
    print(f"multi_count={len(multi)}")
    print(f"units_count={len(units)}")
    print()
    print("=== state_mismatch_funds rows (all) ===")
    print(f"{'games_id':>9} {'tier':>4} {'cop0':>4} {'cop1':>4} {'env':>4} {'day':>4} {'kind':>10} delta")
    for r in sorted(funds, key=lambda x: x.get("games_id", 0)):
        msg = r.get("message", "")
        gid = r.get("games_id")
        print(f"{gid:>9} {r.get('tier'):>4} {r.get('co_p0_id'):>4} {r.get('co_p1_id'):>4} {r.get('approx_envelope_index'):>4} {r.get('approx_day'):>4} {str(r.get('approx_action_kind')):>10} {msg[:160]}")
    print()
    print("=== state_mismatch_multi rows ===")
    for r in sorted(multi, key=lambda x: x.get("games_id", 0)):
        msg = r.get("message", "")
        gid = r.get("games_id")
        print(f"{gid:>9} {r.get('tier'):>4} {r.get('co_p0_id'):>4} {r.get('co_p1_id'):>4} {r.get('approx_envelope_index'):>4} {r.get('approx_day'):>4} {str(r.get('approx_action_kind')):>10} {msg[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
