#!/usr/bin/env python3
"""
Filter and summarize ``logs/game_log.jsonl`` for Phase 1 slice metrics.

Rows are JSON objects separated by blank lines (see ``rl.env._append_game_log_line``).

Examples::

  python tools/slice_game_log.py --curriculum-tag misery-andy
  python tools/slice_game_log.py --learner-seat 1 --reward-mode phi --map-id 123858
  python tools/slice_game_log.py --log-schema 1.9 --export slice.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "logs" / "game_log.jsonl"


def _iter_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def _match(
    rec: dict[str, Any],
    *,
    curriculum_tag: Optional[str],
    learner_seat: Optional[int],
    reward_mode: Optional[str],
    map_id: Optional[int],
    tier: Optional[str],
    log_schema: Optional[str],
    machine_id: Optional[str],
    opponent_sampler: Optional[str],
    arch_version: Optional[str],
) -> bool:
    if curriculum_tag is not None and rec.get("curriculum_tag") != curriculum_tag:
        return False
    if learner_seat is not None:
        ls = rec.get("learner_seat", rec.get("agent_plays", 0))
        try:
            if int(ls) != int(learner_seat):
                return False
        except (TypeError, ValueError):
            return False
    if reward_mode is not None and str(rec.get("reward_mode", "")) != reward_mode:
        return False
    if map_id is not None and rec.get("map_id") != map_id:
        return False
    if tier is not None and str(rec.get("tier", "")) != tier:
        return False
    if log_schema is not None and str(rec.get("log_schema_version", "")) != log_schema:
        return False
    if machine_id is not None:
        mid = rec.get("machine_id")
        if machine_id == "__nonnull__":
            if mid is None:
                return False
        elif str(mid) != machine_id:
            return False
    if opponent_sampler is not None and str(rec.get("opponent_sampler", "")) != opponent_sampler:
        return False
    if arch_version is not None and str(rec.get("arch_version", "")) != arch_version:
        return False
    return True


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}

    turns = [int(r["turns"]) for r in records if r.get("turns") is not None]
    winners = [r.get("winner") for r in records]

    def _learner_seat(r: dict[str, Any]) -> int:
        try:
            return int(r.get("learner_seat", r.get("agent_plays", 0)))
        except (TypeError, ValueError):
            return 0

    learner_wins = 0
    decided = 0
    for r in records:
        w = r.get("winner")
        if w is None:
            continue
        decided += 1
        try:
            if int(w) == _learner_seat(r):
                learner_wins += 1
        except (TypeError, ValueError):
            continue

    trunc = sum(1 for r in records if r.get("truncated"))
    return {
        "n": n,
        "mean_turns": round(statistics.mean(turns), 2) if turns else None,
        "median_turns": statistics.median(turns) if turns else None,
        "truncated_count": trunc,
        "truncated_pct": round(100.0 * trunc / n, 1) if n else 0.0,
        "decided_games": decided,
        "learner_win_rate": round(learner_wins / decided, 4) if decided else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter game_log.jsonl and print summary stats.")
    ap.add_argument("--game-log", type=Path, default=DEFAULT_LOG, help="Path to game_log.jsonl")
    ap.add_argument("--curriculum-tag", type=str, default=None)
    ap.add_argument("--learner-seat", type=int, choices=(0, 1), default=None)
    ap.add_argument("--reward-mode", type=str, default=None, choices=("phi", "level"))
    ap.add_argument("--map-id", type=int, default=None)
    ap.add_argument("--tier", type=str, default=None)
    ap.add_argument("--log-schema", type=str, default=None, help='e.g. "1.9"')
    ap.add_argument("--machine-id", type=str, default=None, help='match exact id, or "__nonnull__"')
    ap.add_argument("--opponent-sampler", type=str, default=None, choices=("uniform", "pfsp"))
    ap.add_argument("--arch-version", type=str, default=None)
    ap.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Write matching records as JSONL (one JSON object per line)",
    )
    args = ap.parse_args()

    if not args.game_log.is_file():
        print(f"Missing file: {args.game_log}")
        return 1

    all_rows, bad = _iter_records(args.game_log)
    filtered = [
        r
        for r in all_rows
        if _match(
            r,
            curriculum_tag=args.curriculum_tag,
            learner_seat=args.learner_seat,
            reward_mode=args.reward_mode,
            map_id=args.map_id,
            tier=args.tier,
            log_schema=args.log_schema,
            machine_id=args.machine_id,
            opponent_sampler=args.opponent_sampler,
            arch_version=args.arch_version,
        )
    ]

    print(f"game_log: {args.game_log.resolve()}")
    print(f"total_parsed_rows: {len(all_rows)}  bad_json_lines: {bad}")
    print(f"matched: {len(filtered)}")
    print("summary:", json.dumps(_summarize(filtered), indent=2))

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("w", encoding="utf-8") as out:
            for r in filtered:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote: {args.export.resolve()} ({len(filtered)} lines)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
