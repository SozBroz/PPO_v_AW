"""Classify state_mismatch_funds and _multi rows from a register."""
import json
import sys
from pathlib import Path


def main(path: str) -> None:
    funds_rows = []
    multi_rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cls = r.get("class")
        if cls == "state_mismatch_funds":
            funds_rows.append(r)
        elif cls == "state_mismatch_multi":
            multi_rows.append(r)

    def delta(r):
        fd = r["state_mismatch"]["diff_summary"].get("funds_delta_by_seat", {})
        return sum(abs(int(v)) for v in fd.values())

    print(f"funds={len(funds_rows)} multi={len(multi_rows)}")
    print("--- FUNDS ---")
    for r in sorted(funds_rows, key=lambda x: -delta(x)):
        fd = r["state_mismatch"]["diff_summary"].get("funds_delta_by_seat", {})
        print(
            f"gid={r['games_id']} co_p0={r['co_p0_id']} co_p1={r['co_p1_id']} "
            f"day={r['approx_day']} env={r['approx_envelope_index']}/{r['envelopes_total']} "
            f"delta={dict(fd)} | {r['message'][:160]}"
        )
    print("--- MULTI ---")
    for r in sorted(multi_rows, key=lambda x: -delta(x)):
        fd = r["state_mismatch"]["diff_summary"].get("funds_delta_by_seat", {})
        print(
            f"gid={r['games_id']} co_p0={r['co_p0_id']} co_p1={r['co_p1_id']} "
            f"day={r['approx_day']} env={r['approx_envelope_index']}/{r['envelopes_total']} "
            f"delta={dict(fd)} | {r['message'][:200]}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
