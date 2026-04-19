#!/usr/bin/env python3
"""
Trend of property holdings at game end from ``data/game_log.jsonl``.

The log does **not** store a per-turn “captured this game” counter — only
``property_count: [p0, p1]`` (buildings owned at termination). We treat:

* **total_owned** — ``p0 + p1`` (both sides’ property counts at end). On a
  fixed map this sum is usually nearly constant; a strong upward *trend* here
  is unexpected unless the map set or schema changed.
* **p0_owned** / **p1_owned** — usual training read: agent (red) vs opponent
  holdings. For P0 training, **p0_owned** trending up is the meaningful signal.

Trend checks (chronological order):

* Mean per time **bin** (first vs last bin).
* **Linear regression** slope vs game index (``statistics.linear_regression``).

Examples::

  python tools/analyze_game_log_property_trend.py
  python tools/analyze_game_log_property_trend.py --since 2026-04-18
  python tools/analyze_game_log_property_trend.py --all-dates --curriculum-tag misery-andy
  python tools/analyze_game_log_property_trend.py --bins 8
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data" / "game_log.jsonl"


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


def _sort_ts(rec: dict[str, Any]) -> float:
    raw = rec.get("timestamp")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    ts_iso = rec.get("timestamp_iso")
    if ts_iso:
        try:
            return datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return 0.0


@dataclass(frozen=True)
class Row:
    total: int
    p0: int
    p1: int
    rec: dict[str, Any]


def _parse_property_row(rec: dict[str, Any]) -> Optional[Row]:
    pc = rec.get("property_count")
    if not isinstance(pc, list) or len(pc) != 2:
        return None
    try:
        p0, p1 = int(pc[0]), int(pc[1])
    except (TypeError, ValueError):
        return None
    if p0 < 0 or p1 < 0:
        return None
    return Row(total=p0 + p1, p0=p0, p1=p1, rec=rec)


def _linear_slope(y: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) for y vs x = 0..n-1 using stdlib."""
    n = len(y)
    if n < 2:
        return float("nan"), float("nan")
    x = tuple(float(i) for i in range(n))
    return statistics.linear_regression(x, tuple(y))


def _pearson_r(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2 or n != len(b):
        return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa = statistics.pstdev(a) if n > 1 else 0.0
    sb = statistics.pstdev(b) if n > 1 else 0.0
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n
    return cov / (sa * sb)


def _bin_means(values: list[float], bins: int) -> list[tuple[int, int, float]]:
    """List of (start_idx, end_idx_exclusive, mean) for equal-sized bins."""
    n = len(values)
    if n == 0 or bins < 1:
        return []
    out: list[tuple[int, int, float]] = []
    base = n // bins
    extra = n % bins
    start = 0
    for b in range(bins):
        sz = base + (1 if b < extra else 0)
        if sz == 0:
            continue
        end = start + sz
        chunk = values[start:end]
        out.append((start, end, statistics.mean(chunk)))
        start = end
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Property-count trend from game_log.jsonl (end-of-game holdings)."
    )
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
    ap.add_argument(
        "--curriculum-tag",
        type=str,
        default=None,
        help="only rows with this curriculum_tag (exact match)",
    )
    ap.add_argument(
        "--map-id",
        type=int,
        default=None,
        help="only rows with this map_id",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=4,
        metavar="N",
        help="number of chronological bins for mean comparison (default: 4)",
    )
    args = ap.parse_args()

    if args.since:
        cutoff = date.fromisoformat(args.since)
    else:
        cutoff = datetime.now().astimezone().date()

    if not args.game_log.is_file():
        print(f"Missing file: {args.game_log}")
        raise SystemExit(1)

    rows: list[Row] = []
    bad_json = 0
    no_ts = 0
    skipped_before_cutoff = 0
    no_pc = 0
    tag_skip = 0
    map_skip = 0

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
            if args.curriculum_tag is not None:
                if rec.get("curriculum_tag") != args.curriculum_tag:
                    tag_skip += 1
                    continue
            if args.map_id is not None:
                if rec.get("map_id") != args.map_id:
                    map_skip += 1
                    continue
            ld = _episode_local_date(rec)
            if ld is None:
                no_ts += 1
                continue
            if not args.all_dates and ld < cutoff:
                skipped_before_cutoff += 1
                continue
            pr = _parse_property_row(rec)
            if pr is None:
                no_pc += 1
                continue
            rows.append(pr)

    rows.sort(key=lambda r: (_sort_ts(r.rec), r.rec.get("game_id", 0)))

    print("game_log:", args.game_log.resolve())
    if args.all_dates:
        print("date filter: none (--all-dates)")
    else:
        print("date filter: local episode date >=", cutoff.isoformat())
    if args.curriculum_tag:
        print("curriculum_tag filter:", args.curriculum_tag)
    if args.map_id is not None:
        print("map_id filter:", args.map_id)
    print(
        "included_games:",
        len(rows),
        "| bad_json:",
        bad_json,
        "| no_timestamp:",
        no_ts,
        "| skipped_before_cutoff:",
        skipped_before_cutoff,
        "| no_property_count:",
        no_pc,
        "| skipped_wrong_tag:",
        tag_skip,
        "| skipped_wrong_map:",
        map_skip,
    )
    print()

    if len(rows) < 2:
        print("Need at least 2 games with property_count for a trend.")
        raise SystemExit(0)

    totals = [float(r.total) for r in rows]
    p0s = [float(r.p0) for r in rows]
    p1s = [float(r.p1) for r in rows]
    idx = [float(i) for i in range(len(rows))]

    def report_metric(name: str, vals: list[float]) -> None:
        slope, intercept = _linear_slope(vals)
        r_idx = _pearson_r(idx, vals)
        print(f"=== {name} ===")
        print(f"  overall mean: {statistics.mean(vals):.3f}  "
              f"stdev: {statistics.pstdev(vals):.3f}  "
              f"min..max: {min(vals):.0f}..{max(vals):.0f}")
        if not math.isnan(slope):
            print(f"  linear trend vs time index: slope={slope:+.5f} per game  "
                  f"(intercept={intercept:.3f})")
        if not math.isnan(r_idx):
            print(f"  Pearson r (index vs {name}): {r_idx:+.3f}")
        bm = _bin_means(vals, min(args.bins, len(vals)))
        if len(bm) >= 2:
            first_m, last_m = bm[0][2], bm[-1][2]
            delta = last_m - first_m
            print(f"  binned means (n_bins={len(bm)}): "
                  f"first_bin={first_m:.3f}  last_bin={last_m:.3f}  "
                  f"delta_last_minus_first={delta:+.3f}")
            if delta > 0.05:
                verdict = "upward"
            elif delta < -0.05:
                verdict = "downward"
            else:
                verdict = "flat"
            print(f"  heuristic (binned): {verdict}")
        print()

    print(
        "Note: 'total' = p0+p1 at game end; on one map it is usually stable.\n"
        "      For P0 training, prefer **p0_owned** trend.\n"
    )
    report_metric("total_owned (p0+p1)", totals)
    report_metric("p0_owned (agent)", p0s)
    report_metric("p1_owned (opponent)", p1s)


if __name__ == "__main__":
    main()
