#!/usr/bin/env python3
"""
Pick the next N Global League **std** catalog games that lack a replay zip under
``replays/amarriner_gl/`` (newest ``games_id`` first), and write ``games_id`` lines
for ``tools/amarriner_download_replays.py --games-ids-file``.

Examples::

  python tools/plan_gl_replay_downloads.py --count 200 --out logs/next_200_games_ids.txt
  python tools/amarriner_download_replays.py --games-ids-file logs/next_200_games_ids.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.amarriner_catalog_cos import catalog_row_has_both_cos  # noqa: E402
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=ROOT / "data" / "amarriner_gl_std_catalog.json")
    ap.add_argument("--map-pool", type=Path, default=ROOT / "data" / "gl_map_pool.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "replays" / "amarriner_gl")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "next_gl_replay_games_ids.txt")
    args = ap.parse_args()

    if not args.catalog.is_file():
        print(f"[plan] missing catalog {args.catalog}", file=sys.stderr)
        return 1
    if not args.map_pool.is_file():
        print(f"[plan] missing map pool {args.map_pool}", file=sys.stderr)
        return 1

    std_ids = gl_std_map_ids(args.map_pool)
    cat = json.loads(args.catalog.read_text(encoding="utf-8"))
    missing: list[int] = []
    for g in (cat.get("games") or {}).values():
        if not isinstance(g, dict):
            continue
        if not catalog_row_has_both_cos(g):
            continue
        mid = g.get("map_id")
        if mid is None or int(mid) not in std_ids:
            continue
        gid = int(g["games_id"])
        dest = args.out_dir / f"{gid}.zip"
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        missing.append(gid)

    missing.sort(reverse=True)
    take = missing[: max(0, args.count)]
    take.sort()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(x) for x in take) + ("\n" if take else ""), encoding="utf-8")

    if take:
        print(
            f"[plan] catalog_std_missing={len(missing)} wrote={len(take)} -> {args.out} "
            f"(games_id range {take[0]}..{take[-1]})"
        )
    else:
        print(f"[plan] catalog_std_missing={len(missing)} wrote=0 -> {args.out}")
    if len(missing) < args.count:
        print(
            f"[plan] WARN: only {len(missing)} games need zips; extend catalog with "
            f"``python tools/amarriner_gl_catalog.py build --first-start <next> --max-pages N``",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
