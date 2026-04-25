#!/usr/bin/env python3
"""Phase 11J-BUILD-NO-OP-CLUSTER-CLOSE — check status of expected gids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SASHA = {1622501, 1624764, 1626284}
TARGETS = {1607045, 1624082, 1627563, 1628849, 1630341, 1632226,
           1632289, 1634961, 1634980, 1635679, 1635846, 1637338}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", type=Path, required=True)
    args = ap.parse_args()
    found: dict[int, str] = {}
    for ln in args.register.open(encoding="utf-8"):
        r = json.loads(ln)
        gid = r.get("games_id")
        if gid in SASHA or gid in TARGETS:
            found[gid] = r.get("class")
    print("Sasha wave-of-five (must stay ok):")
    for g in sorted(SASHA):
        print(f"  {g}: {found.get(g, 'NOT IN BATCH')}")
    print("Build no-op 12 targets:")
    for g in sorted(TARGETS):
        st = found.get(g, "NOT IN BATCH")
        marker = "[CLOSED]" if st == "ok" else "[STILL FAILING]" if st else ""
        print(f"  {g}: {st}  {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
