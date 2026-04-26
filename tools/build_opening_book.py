#!/usr/bin/env python3
"""
Build `ranked_std_human_openings.jsonl` from `opening_demos.jsonl` (one book per
session that lists ordered ``action_idx`` for the book seat).

**Length of the opening** is the number of actions in ``action_indices`` — typically
from **short** replays produced by :mod:`tools.truncate_trace_for_opening` so you do
not rely on calendar metadata or ``--opening-book-days`` in training (use default 0).

Each line sets ``horizon_days`` to **0** (no per-day cap in the opening controller;
only the action list and legality matter).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _book_key_for_session(
    map_id: int, seat: int, co_id: int | None, session_id: str, n_actions: int
) -> str:
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8]
    co = int(co_id) if co_id is not None else -1
    return f"g{map_id}_s{seat}_co{co}_n{n_actions}_{h}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--seat",
        type=int,
        default=1,
        help="Build books for this engine seat (1 = P1 / opponent in default training)",
    )
    ap.add_argument(
        "--dedupe",
        type=str,
        default="action_prefix",
        choices=("action_prefix", "none"),
        help="dedupe: drop books whose action_indices prefix matches an earlier one",
    )
    ap.add_argument(
        "--top-k-per-map",
        type=int,
        default=0,
        dest="top_k_per_map",
        help="If >0, after dedupe cap to K random books per (map_id, seat)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(args.demos, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("active_player", -1)) != int(args.seat):
                continue
            sid = str(row.get("book_session_id") or row.get("session_id", "unknown"))
            by_session[sid].append(row)

    books: list[dict[str, Any]] = []
    for sid, rows in by_session.items():
        rows.sort(key=lambda r: (r.get("trace_index", 0), r.get("awbw_turn", 0), r.get("calendar_turn", 0)))
        if not rows:
            continue
        r0 = rows[0]
        map_id = int(r0.get("map_id", 0) or 0)
        action_indices = [int(r.get("action_idx", 0)) for r in rows]
        n_act = len(action_indices)
        co0 = int(r0.get("co0", 0) or 0)
        co1 = int(r0.get("co1", 0) or 0)
        co_id: int | None = co0 if int(args.seat) == 0 else co1
        if co_id == 0:
            co_id = None
        tier = r0.get("tier")
        tier_s = str(tier) if tier not in (None, "") else None
        book_id = _book_key_for_session(map_id, int(args.seat), co_id, sid, n_act)
        books.append(
            {
                "book_id": book_id,
                "source_game_id": int(r0.get("source_game_id", 0) or 0),
                "map_id": map_id,
                "seat": int(args.seat),
                "co0": co0 if co0 else None,
                "co1": co1 if co1 else None,
                "co_id": co_id,
                "co_name": None,
                "tier": tier_s,
                "horizon_days": 0,
                "opening_player": 0,
                "settings_hash": "unknown",
                "session_id": sid,
                "book_session_id": sid,
                "action_indices": action_indices,
                "validation": {
                    "legal_replay": False,
                    "source": "build_opening_book",
                    "note": "Run tools/validate_opening_book.py before training.",
                },
            }
        )

    if args.dedupe == "action_prefix":
        seen: set[tuple[Any, ...]] = set()
        kept: list[dict[str, Any]] = []
        for b in books:
            k = (b.get("map_id"), b.get("seat"), tuple(b.get("action_indices", ())[:12]))
            if k in seen:
                continue
            seen.add(k)
            kept.append(b)
        books = kept

    if int(args.top_k_per_map) > 0:
        per: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for b in books:
            per[(int(b["map_id"]), int(b["seat"]))].append(b)
        trimmed: list[dict[str, Any]] = []
        for _k, bs in per.items():
            rng.shuffle(bs)
            trimmed.extend(bs[: int(args.top_k_per_map)])
        books = trimmed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for b in books:
            f.write(json.dumps(b) + "\n")
    print(f"[build_opening_book] books={len(books)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
