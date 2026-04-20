"""Emit buggy (non-ok) games from a desync register JSONL. Run from repo root."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--register",
        type=Path,
        default=ROOT / "logs" / "desync_register_20260420_clean.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path (default: print CSV to stdout)",
    )
    ap.add_argument(
        "--exclude-aborted",
        action="store_true",
        help="Omit replay_aborted (resign-only) rows",
    )
    args = ap.parse_args()

    if not args.register.is_file():
        print(f"missing {args.register}", file=sys.stderr)
        return 1

    rows = [
        json.loads(line)
        for line in args.register.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    buggy = [r for r in rows if r.get("class") != "ok"]
    if args.exclude_aborted:
        buggy = [r for r in buggy if r.get("class") != "replay_aborted"]

    by = Counter(r["class"] for r in buggy)
    print(f"source: {args.register}", file=sys.stderr)
    print(f"total games: {len(rows)} | buggy: {len(buggy)}", file=sys.stderr)
    print(f"by class: {dict(by)}", file=sys.stderr)
    print("", file=sys.stderr)

    fieldnames = [
        "games_id",
        "map_id",
        "tier",
        "class",
        "approx_day",
        "actions_applied",
        "message",
        "zip_path",
    ]

    def row_out(r: dict) -> dict:
        msg = (r.get("message") or "").replace("\n", " ").strip()
        base = {k: r.get(k) for k in fieldnames[:-2]}
        base["message"] = msg[:500]
        base["zip_path"] = r.get("zip_path", "")
        return base

    out_rows = [row_out(r) for r in sorted(buggy, key=lambda x: (x["class"], x["games_id"]))]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
