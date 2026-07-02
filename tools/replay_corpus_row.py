#!/usr/bin/env python3
"""Replay one RHEA game corpus JSONL row through the engine."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.game_corpus import replay_corpus_row, state_fingerprint  # noqa: E402


def _load_row(arg: str) -> dict:
    p = Path(arg)
    if p.is_file():
        line = p.read_text(encoding="utf-8").splitlines()[0]
        return json.loads(line)
    return json.loads(arg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "row",
        help="Path to a JSONL file (first line) or inline JSON object",
    )
    ap.add_argument("--map-pool", type=Path, default=ROOT / "data" / "gl_map_pool.json")
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "data" / "maps")
    args = ap.parse_args()

    row = _load_row(args.row)
    final = replay_corpus_row(row, map_pool=args.map_pool, maps_dir=args.maps_dir)
    print(json.dumps(state_fingerprint(final), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
