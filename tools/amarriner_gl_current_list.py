#!/usr/bin/env python3
"""
Scrape in-progress Global League **std** games from ``gamescurrent_all.php``.

Paginates ``start=1, 51, 101, …`` (50 per page) and merges into JSON, same row
shape as ``data/amarriner_gl_current_list_p1.json`` (compatible with
``tools/desync_audit_amarriner_live.py``).

Examples::

  python tools/amarriner_gl_current_list.py build --max-pages 5 --out data/amarriner_gl_current_list_250.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.amarriner_gl_catalog import parse_gamescompleted_listing  # noqa: E402
from tools.amarriner_gl_catalog import _fetch  # noqa: E402

LIST_URL = (
    "https://awbw.amarriner.com/gamescurrent_all.php?start={start}&league=Y&type=std"
)


def build_current_list(
    *,
    out_path: Path,
    first_start: int = 1,
    page_stride: int = 50,
    max_pages: int | None = None,
    sleep_s: float = 0.35,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {
        "source_url_template": LIST_URL,
        "league": "Y",
        "type": "std",
        "page_stride": page_stride,
        "note": "In-progress GL std games (gamescurrent_all.php)",
    }

    pages_done = 0
    start = first_start
    while True:
        if max_pages is not None and pages_done >= max_pages:
            break
        if sleep_s > 0 and pages_done > 0:
            time.sleep(sleep_s)
        url = LIST_URL.format(start=start)
        html = _fetch(url)
        rows = parse_gamescompleted_listing(html)
        pages_done += 1
        if not rows:
            break
        for row in rows:
            row["source_start"] = start
            row["scraped_at"] = datetime.now(timezone.utc).isoformat()
            games[str(row["games_id"])] = row
        if len(rows) < page_stride:
            break
        start += page_stride

    meta["scraped_at"] = datetime.now(timezone.utc).isoformat()
    meta["pages_fetched_this_run"] = pages_done
    meta["n_games"] = len(games)

    payload = {"meta": meta, "games": games}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "amarriner_gl_current_list_250.json")
    ap.add_argument("--first-start", type=int, default=1)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--max-pages", type=int, default=5, help="Default 5 pages ≈ 250 games")
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()

    payload = build_current_list(
        out_path=args.out,
        first_start=args.first_start,
        page_stride=args.stride,
        max_pages=args.max_pages,
        sleep_s=args.sleep,
    )
    print(
        f"[current_list] wrote {args.out} games={payload['meta']['n_games']} "
        f"pages={payload['meta']['pages_fetched_this_run']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
