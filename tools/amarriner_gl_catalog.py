"""
Amarriner Global League **Std** completed-game catalog (HTML only — no replay download).

- Paginates ``gamescompleted.php?league=Y&type=std&start=`` with **start = 50*x + 1**
  (x = 0, 1, 2, …), i.e. 1, 51, 101, … which matches the site’s ~50-games-per-page layout.
- Parses each **listing** row (anchor ``game_<id>``): ``games_id``, GL tier, matchup title,
  ``maps_id``, map link text, and both CO portrait ids (AWBW ``co_id`` scale).
- Writes / merges a **local JSON cache** so you do not re-hit the site unnecessarily.

Replay zips still require a logged-in browser session elsewhere.

Examples::

  python tools/amarriner_gl_catalog.py build --max-pages 3 --sleep 0.35
  python tools/amarriner_gl_catalog.py build   # full crawl until a short page
  python tools/amarriner_gl_catalog.py count --map-id 123858 --co-id 1  # uses cache if present
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tools.amarriner_catalog_cos import catalog_row_has_both_cos

BASE = "https://awbw.amarriner.com"
LIST_URL = BASE + "/gamescompleted.php?league=Y&type=std&start={start}"

# Split listing into per-game HTML chunks (50 games per full page when present).
_RE_GAME_ANCHOR = re.compile(r'<a class="anchor" name="game_(\d+)"></a>', re.I)
# Player portraits in the matchup row (excludes small "CO ban" strip portraits).
# Must match both ``ds/`` (DS-style sheets) and ``aw2/`` (e.g. Sturm, Von Bolt) —
# the old ``.../ds/...``-only pattern yielded ``co_p*_id: null`` whenever a row used aw2.
# Filename stops at ``?`` so ``[^?]+\.png`` cannot swallow ``?v=`` into the name.
_RE_PLAYER_CO = re.compile(
    r"class='co_portrait'\s+src=terrain/co-portraits/[^/]+/[^?]+\.png\?v=(\d+)\?v=",
    re.I,
)
# Map name follows a thumbnail <br>Name</span> inside the prevmaps cell.
_RE_MAP_LINK = re.compile(
    r"prevmaps\.php\?maps_id=(\d+)>.*?<br>\s*([^<]+?)\s*</span>",
    re.I | re.S,
)
_RE_TITLE = re.compile(
    r"<span><b>GL STD \[([^\]]+)\]:([^<]+)</b></span>",
    re.I,
)
_RE_LIST_NUM = re.compile(r"<b>(\d+)\.&nbsp;</b>")
# Game length (final calendar day) in listing metadata: <b>Day N</b> … <b>||</b> … Ended on
_RE_LISTING_FINAL_DAY = re.compile(
    r"<b>\s*Day\s+(\d+)\s*</b>\s*</td>\s*<td[^>]*>\s*<b>\|\|</b>",
    re.I | re.S,
)


def _fetch(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AWBW-catalog/1.0; +local research)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_gamescompleted_listing(html: str) -> list[dict[str, Any]]:
    """
    Parse one ``gamescompleted.php`` HTML page into game records (no per-game HTTP).
    """
    matches = list(_RE_GAME_ANCHOR.finditer(html))
    out: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        gid = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(html)
        chunk = html[start:end]

        title_m = _RE_TITLE.search(chunk)
        tier = title_m.group(1).strip() if title_m else None
        matchup = title_m.group(2).strip() if title_m else None

        map_m = _RE_MAP_LINK.search(chunk)
        map_id = int(map_m.group(1)) if map_m else None
        map_name = map_m.group(2).strip() if map_m else None

        co_ids = [int(x) for x in _RE_PLAYER_CO.findall(chunk)]
        co_p0 = co_ids[0] if len(co_ids) > 0 else None
        co_p1 = co_ids[1] if len(co_ids) > 1 else None

        list_m = _RE_LIST_NUM.search(chunk)
        list_index = int(list_m.group(1)) if list_m else None

        fd_m = _RE_LISTING_FINAL_DAY.search(chunk)
        listing_final_day = int(fd_m.group(1)) if fd_m else None

        out.append(
            {
                "games_id": gid,
                "tier": tier,
                "matchup": matchup,
                "map_id": map_id,
                "map_name": map_name,
                "co_p0_id": co_p0,
                "co_p1_id": co_p1,
                "list_index": list_index,
                "listing_final_day": listing_final_day,
            }
        )
    return out


def iter_listing_starts(first_start: int = 1, page_stride: int = 50) -> Iterator[int]:
    """Yield start=1, 51, 101, …"""
    s = first_start
    while True:
        yield s
        s += page_stride


def build_catalog(
    *,
    out_path: Path,
    first_start: int = 1,
    page_stride: int = 50,
    max_pages: Optional[int] = None,
    sleep_s: float = 0.25,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Fetch listing pages, merge into *out_path* JSON (dict keyed by ``games_id``).

    Stops when a page yields **zero** games, or fewer than *page_stride* games
    (last page), or *max_pages* is reached.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {
        "source_url_template": LIST_URL,
        "league": "Y",
        "type": "std",
        "page_stride": page_stride,
    }
    if resume and out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            games = {str(k): v for k, v in prev.get("games", {}).items()}
            meta.update({k: v for k, v in prev.get("meta", {}).items() if k in ("first_crawl_at",)})
        except json.JSONDecodeError:
            games = {}

    pages_done = 0
    for start in iter_listing_starts(first_start, page_stride):
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
        # Short final page
        if len(rows) < page_stride:
            break

    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta["pages_fetched_this_run"] = pages_done
    meta["n_games"] = len(games)
    if "first_crawl_at" not in meta:
        meta["first_crawl_at"] = meta["updated_at"]

    payload = {"meta": meta, "games": games}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_catalog(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def count_from_catalog(
    catalog: dict[str, Any],
    *,
    map_id: int,
    co_id: int,
) -> list[dict[str, Any]]:
    """Filter cached ``games`` for map + both CO ids == *co_id*."""
    out = []
    for g in catalog.get("games", {}).values():
        if g.get("map_id") != map_id:
            continue
        if g.get("co_p0_id") == co_id and g.get("co_p1_id") == co_id:
            out.append(g)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    default_out = Path(__file__).resolve().parents[1] / "data" / "amarriner_gl_std_catalog.json"

    b = sub.add_parser("build", help="Paginate listing (start=50x+1) and write/merge JSON cache")
    b.add_argument("--out", type=Path, default=default_out, help="Output JSON path")
    b.add_argument("--first-start", type=int, default=1, help="First ``start=`` (default 1)")
    b.add_argument("--stride", type=int, default=50, help="Increment for ``start`` (default 50)")
    b.add_argument("--max-pages", type=int, default=None, help="Cap pages this run (default: until short page)")
    b.add_argument("--sleep", type=float, default=0.25, help="Delay between listing page fetches")
    b.add_argument("--no-resume", action="store_true", help="Ignore existing output file (overwrite merge dict)")

    c = sub.add_parser("count", help="Count mirror matchups (optionally from cache)")
    c.add_argument("--map-id", type=int, default=123858)
    c.add_argument("--co-id", type=int, default=1, help="Both players' AWBW co_id (e.g. Andy = 1)")
    c.add_argument("--catalog", type=Path, default=default_out, help="Catalog JSON from ``build``")

    li = sub.add_parser(
        "list-incomplete-cos",
        help="List games whose catalog row is missing co_p0_id or co_p1_id (cannot run engine)",
    )
    li.add_argument("--catalog", type=Path, default=default_out, help="Catalog JSON from ``build``")

    args = ap.parse_args()
    if args.cmd == "build":
        payload = build_catalog(
            out_path=args.out,
            first_start=args.first_start,
            page_stride=args.stride,
            max_pages=args.max_pages,
            sleep_s=args.sleep,
            resume=not args.no_resume,
        )
        print(
            f"[amarriner_gl_catalog] wrote {args.out} "
            f"games={payload['meta']['n_games']} pages_this_run={payload['meta']['pages_fetched_this_run']}"
        )
        return 0

    if args.cmd == "count":
        if not args.catalog.is_file():
            print(f"[amarriner_gl_catalog] no catalog at {args.catalog} — run ``build`` first.")
            return 2
        cat = load_catalog(args.catalog)
        hits = count_from_catalog(cat, map_id=args.map_id, co_id=args.co_id)
        print(f"[amarriner_gl_catalog] catalog games={cat['meta'].get('n_games', '?')}")
        print(f"  map={args.map_id} co_mirror={args.co_id}  matches={len(hits)}")
        for g in hits[:40]:
            print(
                f"    games_id={g['games_id']} tier={g.get('tier')!r} "
                f"co=({g.get('co_p0_id')},{g.get('co_p1_id')}) map={g.get('map_name')!r}"
            )
        if len(hits) > 40:
            print(f"    ... +{len(hits) - 40} more")
        return 0

    if args.cmd == "list-incomplete-cos":
        if not args.catalog.is_file():
            print(f"[amarriner_gl_catalog] no catalog at {args.catalog} — run ``build`` first.")
            return 2
        cat = load_catalog(args.catalog)
        bad: list[dict[str, Any]] = []
        for g in cat.get("games", {}).values():
            if isinstance(g, dict) and not catalog_row_has_both_cos(g):
                bad.append(g)
        bad.sort(key=lambda x: int(x["games_id"]))
        print(
            f"[amarriner_gl_catalog] games missing one or both CO ids: {len(bad)} "
            f"(of {cat['meta'].get('n_games', '?')} total)"
        )
        for g in bad:
            print(
                f"  games_id={g['games_id']} co=({g.get('co_p0_id')!r},{g.get('co_p1_id')!r}) "
                f"tier={g.get('tier')!r} map_id={g.get('map_id')}"
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
