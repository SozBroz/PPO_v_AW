"""
Compare two ``desync_audit.py`` registers (JSONL) and report regression status.

Reads ``BEFORE`` and ``AFTER`` register files (each row is one ``games_id``
audit result with a ``class`` field per ``docs/desync_audit.md``) and prints
three sets:

* ``regressions`` — ids that were ``class == "ok"`` in BEFORE but are
  non-ok in AFTER. **Must be empty** for a fix to be accepted.
* ``fixed`` — ids that were non-ok in BEFORE and are ``ok`` in AFTER.
* ``class_drift`` — ids that were non-ok in both but moved between
  non-ok classes (e.g. ``oracle_gap`` → ``engine_bug``); reviewer must
  decide whether the drift is intentional.

Exit code is ``0`` only if ``regressions`` is empty (so the script can
gate ``desync_audit`` reruns and pre-commit hooks). Counts and per-class
totals are also reported for quick eyeball verification.

Examples::

  python tools/desync_register_diff.py \\
      logs/desync_register_validate_20260420_full.jsonl \\
      logs/desync_register_post_agent1_step1.jsonl

  python tools/desync_register_diff.py BEFORE.jsonl AFTER.jsonl --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_register(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gid = int(r["games_id"])
            rows[gid] = r
    return rows


def _is_ok(row: dict[str, Any]) -> bool:
    return (row.get("class") or row.get("cls")) == "ok"


def _cls(row: dict[str, Any]) -> str:
    return str(row.get("class") or row.get("cls") or "")


def diff_registers(
    before: dict[int, dict[str, Any]],
    after: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    common = set(before) & set(after)
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))

    regressions: list[dict[str, Any]] = []
    fixed: list[dict[str, Any]] = []
    class_drift: list[dict[str, Any]] = []

    for gid in sorted(common):
        b, a = before[gid], after[gid]
        b_ok, a_ok = _is_ok(b), _is_ok(a)
        if b_ok and not a_ok:
            regressions.append({
                "games_id": gid,
                "after_class": _cls(a),
                "after_message": (a.get("message") or "")[:200],
                "approx_day": a.get("approx_day"),
                "approx_action_kind": a.get("approx_action_kind"),
            })
        elif not b_ok and a_ok:
            fixed.append({
                "games_id": gid,
                "before_class": _cls(b),
            })
        elif not b_ok and not a_ok and _cls(b) != _cls(a):
            class_drift.append({
                "games_id": gid,
                "before_class": _cls(b),
                "after_class": _cls(a),
                "after_message": (a.get("message") or "")[:200],
            })

    before_counts = Counter(_cls(r) for r in before.values())
    after_counts = Counter(_cls(r) for r in after.values())

    return {
        "before_count": len(before),
        "after_count": len(after),
        "before_ok": before_counts.get("ok", 0),
        "after_ok": after_counts.get("ok", 0),
        "before_classes": dict(before_counts),
        "after_classes": dict(after_counts),
        "regressions": regressions,
        "fixed": fixed,
        "class_drift": class_drift,
        "only_in_before": only_before,
        "only_in_after": only_after,
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"BEFORE rows: {report['before_count']}  ok={report['before_ok']}")
    print(f"AFTER  rows: {report['after_count']}  ok={report['after_ok']}")
    print()
    print("BEFORE class counts:")
    for k, v in sorted(report["before_classes"].items()):
        print(f"  {k:<28} {v:>4}")
    print("AFTER class counts:")
    for k, v in sorted(report["after_classes"].items()):
        print(f"  {k:<28} {v:>4}")
    print()
    regs = report["regressions"]
    fixed = report["fixed"]
    drift = report["class_drift"]
    only_b = report["only_in_before"]
    only_a = report["only_in_after"]

    print(f"regressions (ok -> non-ok): {len(regs)}")
    for r in regs:
        print(
            f"  REGRESSION gid={r['games_id']} -> {r['after_class']}"
            f" day~{r['approx_day']} kind={r['approx_action_kind']}"
        )
        print(f"      msg: {r['after_message']}")
    print()
    print(f"fixed (non-ok -> ok): {len(fixed)}")
    for r in fixed:
        print(f"  FIXED gid={r['games_id']} (was {r['before_class']})")
    print()
    print(f"class_drift (non-ok -> different non-ok class): {len(drift)}")
    for r in drift:
        print(
            f"  DRIFT gid={r['games_id']} {r['before_class']} -> {r['after_class']}"
        )
        print(f"      msg: {r['after_message']}")
    print()
    if only_b:
        print(f"only in BEFORE ({len(only_b)}): {only_b[:20]}{' ...' if len(only_b) > 20 else ''}")
    if only_a:
        print(f"only in AFTER  ({len(only_a)}): {only_a[:20]}{' ...' if len(only_a) > 20 else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", type=Path, help="baseline register JSONL")
    ap.add_argument("after", type=Path, help="post-change register JSONL")
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit the full report as JSON (regressions / fixed / class_drift / counts)",
    )
    ap.add_argument(
        "--allow-only-in-before",
        action="store_true",
        help=(
            "Do not treat games_ids missing from AFTER as regressions. Use when "
            "scuffed RV1 zips were intentionally deleted (Mandatory closure column 1)."
        ),
    )
    args = ap.parse_args()

    if not args.before.is_file():
        print(f"[desync_register_diff] missing BEFORE: {args.before}", file=sys.stderr)
        return 2
    if not args.after.is_file():
        print(f"[desync_register_diff] missing AFTER: {args.after}", file=sys.stderr)
        return 2

    before = _load_register(args.before)
    after = _load_register(args.after)
    report = diff_registers(before, after)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)

    only_in_before = report["only_in_before"]
    if only_in_before and not args.allow_only_in_before:
        only_b_were_ok = [
            gid for gid in only_in_before if _is_ok(before[gid])
        ]
        if only_b_were_ok:
            print(
                f"[desync_register_diff] FAIL: {len(only_b_were_ok)} ok games dropped"
                f" from AFTER (use --allow-only-in-before only if you intentionally"
                f" deleted scuffed zips). first 10: {only_b_were_ok[:10]}",
                file=sys.stderr,
            )
            return 1

    if report["regressions"]:
        print(
            f"[desync_register_diff] FAIL: {len(report['regressions'])} regressions",
            file=sys.stderr,
        )
        return 1
    print("[desync_register_diff] OK: no regressions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
