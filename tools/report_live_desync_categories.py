#!/usr/bin/env python3
"""
Summarize a live desync register (JSONL): class counts, failure categories, tables.

Examples::

  python tools/report_live_desync_categories.py \\
    --register logs/desync_register_amarriner_live_250.jsonl \\
    --out-md logs/live_250_desync_categories.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)

from tools.live_desync_buckets import BUCKET_TITLE, bucket_live_failure  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=None, help="Optional; noted in markdown meta")
    ap.add_argument("--out-md", type=Path, default=ROOT / "logs" / "live_desync_categories.md")
    ap.add_argument(
        "--out-cleanliness",
        type=Path,
        default=None,
        help="Optional JSON summary (class_counts, first_divergence count)",
    )
    args = ap.parse_args()

    if not args.register.is_file():
        print(f"[report] missing {args.register}", file=sys.stderr)
        return 1

    rows = _load_jsonl(args.register)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    class_c: Counter[str] = Counter()
    for r in rows:
        class_c[r.get("class") or "unknown"] += 1

    failures = [
        r
        for r in rows
        if (r.get("class") or "") != "ok" and r.get("status") == "first_divergence"
    ]

    bucket_c: Counter[str] = Counter()
    for r in failures:
        bid = bucket_live_failure(
            str(r.get("message") or ""),
            str(r.get("class") or ""),
            r.get("approx_action_kind"),
        )
        bucket_c[bid] += 1

    lines: list[str] = []
    lines.append("# Live desync category report")
    lines.append("")
    lines.append(f"**Generated:** {stamp}")
    lines.append(f"**Register:** `{_rel(args.register)}`")
    if args.catalog and args.catalog.is_file():
        lines.append(f"**Catalog:** `{_rel(args.catalog)}`")
    lines.append(f"**Rows:** {len(rows)} (one per audited game)")
    if args.catalog and args.catalog.is_file():
        try:
            cmeta = json.loads(args.catalog.read_text(encoding="utf-8")).get("meta") or {}
            ng = cmeta.get("n_games")
            if ng is not None and int(ng) != len(rows):
                lines.append(
                    f"**Catalog note:** `n_games={ng}` in catalog JSON; "
                    f"{int(ng) - len(rows)} rows omitted by live audit (not GL std map and/or missing CO ids)."
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|-------|------:|")
    for k in sorted(class_c.keys()):
        lines.append(f"| `{k}` | {class_c[k]} |")
    lines.append("")
    n_ok = class_c.get("ok", 0)
    n_fail = len(failures)
    lines.append(
        f"**Pass rate:** {n_ok}/{len(rows)} ({100.0 * n_ok / len(rows):.1f}%) — "
        f"**first divergences:** {n_fail}"
    )
    lines.append("")
    lines.append("## Failure categories (thematic)")
    lines.append("")
    lines.append("| category | count | description |")
    lines.append("|----------|------:|-------------|")
    for bid in sorted(bucket_c.keys(), key=lambda x: (-bucket_c[x], x)):
        title = BUCKET_TITLE.get(bid, bid)
        lines.append(f"| `{bid}` | {bucket_c[bid]} | {title} |")
    lines.append("")

    lines.append("## Failures by category (detail)")
    lines.append("")
    by_b: dict[str, list[dict[str, Any]]] = {}
    for r in failures:
        bid = bucket_live_failure(
            str(r.get("message") or ""),
            str(r.get("class") or ""),
            r.get("approx_action_kind"),
        )
        by_b.setdefault(bid, []).append(r)

    for bid in sorted(by_b.keys(), key=lambda x: (-len(by_b[x]), x)):
        lines.append(f"### {BUCKET_TITLE.get(bid, bid)} (`{bid}`)")
        lines.append("")
        lines.append("| games_id | action | envelope | day | message (head) |")
        lines.append("|----------|--------|----------|-----|----------------|")
        for r in sorted(by_b[bid], key=lambda x: int(x["games_id"])):
            gid = r.get("games_id")
            ak = r.get("approx_action_kind") or "?"
            env = r.get("approx_envelope_index")
            day = r.get("approx_day")
            msg = str(r.get("message") or "").replace("\n", " ")[:120]
            lines.append(f"| {gid} | {ak} | {env} | {day} | {msg} |")
        lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.out_cleanliness:
        summary = {
            "as_of_utc": stamp,
            "register": _rel(args.register),
            "games_audited": len(rows),
            "class_counts": dict(sorted(class_c.items())),
            "first_divergence_total": n_fail,
            "failure_bucket_counts": {k: bucket_c[k] for k in sorted(bucket_c.keys())},
        }
        if args.catalog:
            summary["catalog"] = _rel(args.catalog)
        args.out_cleanliness.parent.mkdir(parents=True, exist_ok=True)
        args.out_cleanliness.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"[report] wrote {args.out_md} rows={len(rows)} failures={n_fail}")
    if args.out_cleanliness:
        print(f"[report] wrote {args.out_cleanliness}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
