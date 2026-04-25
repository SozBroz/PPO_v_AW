"""Print first-drift and failing-build summary for the 4 target GIDs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "logs" / "phase11j_funds_drill.json")
                      .read_text(encoding="utf-8"))
    for c in data["cases"]:
        gid = c["gid"]
        print(f"\n=== {gid} | {c.get('matchup')} | "
              f"co_p0={c.get('co_p0')} co_p1={c.get('co_p1')} | "
              f"map={c.get('map_id')} tier={c.get('tier')} ===")
        rows = c.get("per_envelope") or []
        first_drift = None
        fail_row = None
        for r in rows:
            d = r.get("delta_engine_minus_php")
            if d and any(v != 0 for v in d.values()) and first_drift is None:
                first_drift = r
            if "fail_msg" in r:
                fail_row = r
        if first_drift:
            print(f"  First drift  : env_i={first_drift['env_i']} "
                  f"day={first_drift['day']} pid={first_drift['pid']}")
            print(f"    engine_funds = {first_drift['engine_funds']}")
            print(f"    php_funds    = {first_drift['php_funds']}")
            print(f"    delta(e-p)   = {first_drift['delta_engine_minus_php']}")
            print(f"    eng_props    = {first_drift['engine_props']}")
        else:
            print("  No funds drift observed before failure.")
        if fail_row:
            print(f"  Build no-op  : env_i={fail_row['env_i']} "
                  f"day={fail_row['day']} pid={fail_row['pid']}")
            print(f"    {fail_row['fail_msg']}")
            print(f"    php_funds_pre_env  = {fail_row.get('php_funds_pre_env')}")
            print(f"    engine_funds       = {fail_row.get('engine_funds')}")


if __name__ == "__main__":
    main()
