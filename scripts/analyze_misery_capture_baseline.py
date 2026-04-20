#!/usr/bin/env python3
"""
Slice ``logs/game_log.jsonl`` for Misery (map_id 123858) + Andy mirror (CO id 1 vs 1).

Reports property_count sums, win_condition mix, turns, opponent_type — baseline
metrics for capture-mastery experiments (see Cursor plan
``misery_andy_capture_mastery_*.plan.md``).

Usage:
  python scripts/analyze_misery_capture_baseline.py
  python scripts/analyze_misery_capture_baseline.py path/to/custom.jsonl
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "logs" / "game_log.jsonl"

MISERY_ID = 123858
ANDY_ID = 1


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    if not path.is_file():
        print(f"No file at {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    for part in text.split("\n\n"):
        part = part.strip()
        if not part:
            continue
        try:
            rows.append(json.loads(part))
            continue
        except json.JSONDecodeError:
            pass
        for line in part.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def is_slice(rec: dict) -> bool:
        if rec.get("map_id") != MISERY_ID:
            return False
        if rec.get("p0_co_id") != ANDY_ID or rec.get("p1_co_id") != ANDY_ID:
            return False
        return True

    sl = [r for r in rows if is_slice(r)]
    print(f"Total JSON records: {len(rows)}")
    print(f"Misery ({MISERY_ID}) + Andy mirror (p0_co_id=p1_co_id={ANDY_ID}): {len(sl)}")
    if not sl:
        return

    combined = []
    for r in sl:
        pc = r.get("property_count")
        if isinstance(pc, list) and len(pc) >= 2:
            combined.append(int(pc[0]) + int(pc[1]))

    turns = [int(r.get("turns", 0)) for r in sl]
    wins = Counter(str(r.get("win_condition")) for r in sl)
    opp = Counter(str(r.get("opponent_type", "?")) for r in sl)

    print("\n--- property_count[0]+[1] at game end ---")
    if combined:
        print(f"  n={len(combined)}  median={statistics.median(combined):.1f}  "
              f"mean={statistics.mean(combined):.2f}  min={min(combined)}  max={max(combined)}")

    print("\n--- win_condition ---")
    for k, v in wins.most_common():
        print(f"  {k!r}: {v}")

    print("\n--- turns ---")
    if turns:
        print(f"  median={statistics.median(turns):.1f}  mean={statistics.mean(turns):.2f}")

    print("\n--- opponent_type ---")
    for k, v in opp.most_common():
        print(f"  {k!r}: {v}")

    fc = [r.get("first_p0_capture_p0_step") for r in sl if r.get("first_p0_capture_p0_step") is not None]
    if fc:
        nums = [int(x) for x in fc]
        print("\n--- first_p0_capture_p0_step (when logged) ---")
        print(f"  n={len(nums)}  median={statistics.median(nums):.1f}")


if __name__ == "__main__":
    main()
