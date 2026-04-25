#!/usr/bin/env python3
"""Phase 11J-BUILD-NO-OP-CLUSTER-CLOSE — summarize drill output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "logs" / "phase11j_buildnoop12_drill.json"
CO_DATA = ROOT / "data" / "co_data.json"


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    co_names = json.loads(CO_DATA.read_text(encoding="utf-8"))["cos"]

    def nm(i):
        return co_names.get(str(i), {}).get("name", "?")

    rows_summary = []
    for c in data["cases"]:
        gid = c["gid"]
        p0, p1 = c.get("co_p0"), c.get("co_p1")
        rows = c.get("per_envelope") or []
        fail = next((r for r in rows if "fail_msg" in r), None)
        if not fail:
            print(f"{gid} {nm(p0)} vs {nm(p1)} - NO FAIL")
            continue
        first_drift = None
        for r in rows:
            d = r.get("delta_engine_minus_php")
            if d and any(v != 0 for v in d.values()):
                first_drift = r
                break
        pre = fail.get("php_funds_pre_env", {}) or {}
        eng = fail.get("engine_funds", {}) or {}
        # JSON dict keys come back as strings
        def _g(d, k):
            if k in d: return d[k]
            return d.get(str(k), 0)
        drift = {p: _g(eng, p) - _g(pre, p) for p in (0, 1)}

        print(f"gid={gid} | P0={nm(p0)} P1={nm(p1)}")
        print(f"  fail env={fail['env_i']} day={fail['day']} pid={fail['pid']} act_idx={fail['fail_at_action_idx']}")
        print(f"  msg:  {fail['fail_msg'][:130]}")
        print(f"  pre-env php_funds={pre}  engine_funds={eng}  drift(eng-php)={drift}")
        if first_drift:
            print(f"  first drift env={first_drift['env_i']} day={first_drift['day']} pid={first_drift['pid']}")
            print(f"    delta={first_drift['delta_engine_minus_php']}  eng={first_drift['engine_funds']} php={first_drift['php_funds']}")
        else:
            print("  (no recorded drift before fail)")
        print()
        rows_summary.append({
            "gid": gid, "p0": nm(p0), "p1": nm(p1),
            "fail_env": fail["env_i"], "day": fail["day"],
            "drift_at_fail": drift,
            "first_drift_env": first_drift["env_i"] if first_drift else None,
            "first_drift_day": first_drift["day"] if first_drift else None,
            "first_drift_delta": first_drift["delta_engine_minus_php"] if first_drift else None,
        })

    print("=" * 80)
    print("CLUSTER VIEW")
    print("=" * 80)
    for r in sorted(rows_summary, key=lambda x: (x["p0"], x["p1"], x["gid"])):
        d = r["drift_at_fail"]
        print(f"  {r['gid']:>8} {r['p0']:>8}/{r['p1']:<8} fail_day={r['day']:>3} "
              f"first_drift_day={r['first_drift_day']} drift_at_fail={d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
