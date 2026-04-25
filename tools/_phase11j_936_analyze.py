#!/usr/bin/env python3
"""One-off: parse Phase 11J combined register and print summary stats (read-only)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "logs" / "desync_register_post_phase11j_combined.jsonl"
STD_CAT = ROOT / "data" / "amarriner_gl_std_catalog.json"
EXT_CAT = ROOT / "data" / "amarriner_gl_extras_catalog.json"


def load_games_ids(path: Path) -> set[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    games = raw.get("games") or {}
    out: set[int] = set()
    for g in games.values():
        if isinstance(g, dict) and "games_id" in g:
            out.add(int(g["games_id"]))
    return out


def classify_engine_bug_family(msg: str) -> str:
    """Phase 11D-style F1–F5 heuristic labels (see phase11d_residual_engine_bug_triage.md)."""
    lower = msg.lower()
    if "friendly fire" in lower or "self-target" in lower:
        return "F4"
    if "black_boat" in lower or "unarmed" in lower:
        return "F5"
    if any(k in lower for k in ("super power", "cop ", " co power", "activated power")):
        return "F3"
    if "illegal move" in lower and "not reachable" in lower:
        return "F2"
    if "unit_pos" in lower or "drift" in lower:
        return "F1"
    return "OTHER"


def oracle_gap_bucket(msg: str) -> str:
    if msg.startswith("Move:") and "truncated path" in msg:
        return "Move: truncated path vs AWBW path end (upstream drift)"
    if msg.startswith("Build no-op"):
        return "Build no-op (BUILD refused)"
    if msg.startswith("Fire:"):
        return msg.split(";", 1)[0][:120]
    if msg.startswith("Move:"):
        return "Move: other"
    return msg[:100]


def main() -> None:
    rows: list[dict] = []
    for line in COMBINED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    print(f"games: {len(rows)}")
    std_ids = load_games_ids(STD_CAT)
    ext_ids = load_games_ids(EXT_CAT)
    by_all = Counter(r["class"] for r in rows)
    std_rows = [r for r in rows if r["games_id"] in std_ids]
    ext_rows = [r for r in rows if r["games_id"] in ext_ids]
    print("all classes:", dict(by_all))
    print("std (741)", Counter(r["class"] for r in std_rows))
    print("extras (195)", Counter(r["class"] for r in ext_rows))
    print("engine_bug families:", Counter(classify_engine_bug_family(r["message"]) for r in rows if r["class"] == "engine_bug"))
    print("oracle_gap buckets:", Counter(oracle_gap_bucket(r["message"]) for r in rows if r["class"] == "oracle_gap"))


if __name__ == "__main__":
    main()
