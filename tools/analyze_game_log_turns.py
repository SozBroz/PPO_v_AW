#!/usr/bin/env python3
"""
Drill-down stats on ``turns`` (engine day counter at episode end) from ``logs/game_log.jsonl``.

Default: only games that finished on or after **today** (local calendar), excluding older runs.
Stratifies by ``opponent_type`` and ``curriculum_tag``.

Examples::

  python tools/analyze_game_log_turns.py
  python tools/analyze_game_log_turns.py --since 2026-04-18
  python tools/analyze_game_log_turns.py --all-dates   # no date filter
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "logs" / "game_log.jsonl"


def _episode_local_date(rec: dict[str, Any]) -> Optional[date]:
    ts_iso = rec.get("timestamp_iso")
    if ts_iso:
        try:
            d = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
            return d.astimezone().date()
        except ValueError:
            pass
    raw = rec.get("timestamp")
    if raw is not None:
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).astimezone().date()
        except (TypeError, ValueError, OSError):
            pass
    return None


def _percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _stats(turns: list[int], wins: list[Optional[str]]) -> dict[str, Any]:
    if not turns:
        return {"n": 0}
    s = sorted(turns)
    n = len(s)
    ge100 = sum(1 for x in s if x >= 100)
    eq1 = sum(1 for x in s if x == 1)
    wc = Counter(w for w in wins if w)
    return {
        "n": n,
        "mean": round(statistics.mean(s), 2),
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "p90": round(_percentile(s, 90), 2),
        "p95": round(_percentile(s, 95), 2),
        "pct_turns_ge_100": round(100.0 * ge100 / n, 1),
        "pct_turns_eq_1": round(100.0 * eq1 / n, 1),
        "win_condition_top": wc.most_common(5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze turns in game_log.jsonl by opponent/tag.")
    ap.add_argument(
        "--game-log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"path to JSONL (default: {DEFAULT_LOG})",
    )
    ap.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="include only episodes with local date >= this day (default: today local)",
    )
    ap.add_argument(
        "--all-dates",
        action="store_true",
        help="do not filter by date",
    )
    args = ap.parse_args()

    if args.since:
        cutoff = date.fromisoformat(args.since)
    else:
        cutoff = datetime.now().astimezone().date()

    if not args.game_log.is_file():
        print(f"Missing file: {args.game_log}")
        raise SystemExit(1)

    rows: list[dict[str, Any]] = []
    bad_json = 0
    no_ts = 0
    skipped_before_cutoff = 0

    with args.game_log.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if rec.get("turns") is None:
                continue
            ld = _episode_local_date(rec)
            if ld is None:
                no_ts += 1
                continue
            if not args.all_dates and ld < cutoff:
                skipped_before_cutoff += 1
                continue
            rows.append(rec)

    # Group (opponent_type, curriculum_tag)
    by_ot: dict[str, list[int]] = defaultdict(list)
    by_ot_wc: dict[str, list[Optional[str]]] = defaultdict(list)
    by_tag: dict[str, list[int]] = defaultdict(list)

    for rec in rows:
        ot = str(rec.get("opponent_type") or "missing")
        tag = rec.get("curriculum_tag")
        tag_k = str(tag) if tag else "(none)"
        t = int(rec["turns"])
        wr = rec.get("win_condition")
        by_ot[ot].append(t)
        by_ot_wc[ot].append(wr if isinstance(wr, str) else None)
        by_tag[tag_k].append(t)

    print("game_log:", args.game_log.resolve())
    if args.all_dates:
        print("date filter: none (--all-dates)")
    else:
        print("date filter: local episode date >=", cutoff.isoformat())
    print(
        "included_rows:",
        len(rows),
        "| bad_json_lines:",
        bad_json,
        "| no_usable_timestamp:",
        no_ts,
        "| skipped_before_cutoff:",
        skipped_before_cutoff,
    )
    print()

    print("=== By opponent_type ===")
    for ot in sorted(by_ot.keys(), key=lambda k: (-len(by_ot[k]), k)):
        st = _stats(by_ot[ot], by_ot_wc[ot])
        print(f"\n[{ot}]")
        if st["n"] == 0:
            print("  (no rows)")
            continue
        print(f"  n={st['n']}  mean={st['mean']}  median={st['median']}  "
              f"min={st['min']}  max={st['max']}  p90={st['p90']}  p95={st['p95']}")
        print(f"  pct turns>=100: {st['pct_turns_ge_100']}%  pct turns==1: {st['pct_turns_eq_1']}%")
        if st.get("win_condition_top"):
            print(f"  win_condition: {st['win_condition_top']}")

    print("\n=== By curriculum_tag ===")
    for tag in sorted(by_tag.keys(), key=lambda k: (-len(by_tag[k]), k)):
        st = _stats(by_tag[tag], [])
        print(f"\n[{tag}]")
        if st["n"] == 0:
            continue
        print(f"  n={st['n']}  mean={st['mean']}  median={st['median']}  "
              f"min={st['min']}  max={st['max']}  p90={st['p90']}  p95={st['p95']}")
        print(f"  pct turns>=100: {st['pct_turns_ge_100']}%  pct turns==1: {st['pct_turns_eq_1']}%")


if __name__ == "__main__":
    main()
