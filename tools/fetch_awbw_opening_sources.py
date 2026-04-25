#!/usr/bin/env python3
"""
Fetch replay zips and build a manifest for elite opening-book / BC pipelines.

- **manual_games:** download ``replay_download.php?games_id=`` (requires
  ``secrets.txt`` login, same as ``tools/amarriner_download_replays.py``).
- **completed_game_queries:** crawl ``gamescompleted.php?league=Y&type=std``
  (see ``tools/amarriner_gl_catalog.py``), keep rows whose ``matchup`` contains
  the requested username and whose ``map_id`` is in the GL **std** pool.

Outputs under ``--out-dir``::

  games/<games_id>.zip
  completed/<username>_page_<n>.html   (optional debug)
  manifest.jsonl

Example::

  python tools/fetch_awbw_opening_sources.py \\
    --sources data/human_openings/elite_opening_sources.yaml \\
    --map-pool data/gl_map_pool.json \\
    --out-dir data/human_openings/raw \\
    --respect-cache \\
    --sleep-s 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.amarriner_download_replays import (  # noqa: E402
    BASE_URL,
    HEADERS,
    LOGIN_URL,
    _download_replay_zip,
)
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402

SECRETS = ROOT / "secrets.txt"
LIST_URL = BASE_URL + "/gamescompleted.php?league=Y&type=std&start={start}"


def _login(session: requests.Session, username: str, password: str) -> bool:
    r = session.get(LOGIN_URL, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "lxml")
    payload: dict[str, str] = {}
    form = soup.find("form")
    if form:
        for inp in form.find_all("input"):
            name = inp.get("name")
            if name:
                payload[name] = inp.get("value", "")
    payload["username"] = username
    payload["password"] = password
    r2 = session.post(
        LOGIN_URL, data=payload, headers=HEADERS, timeout=20, allow_redirects=True
    )
    return "logout" in r2.text.lower() or username.lower() in r2.text.lower()


def _fetch_list_page(start: int) -> str:
    import urllib.request

    url = LIST_URL.format(start=start)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": HEADERS.get("User-Agent", "AWBW/1.0")},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read().decode("utf-8", "replace")


def _parse_gamescompleted_rows(html: str) -> list[dict[str, Any]]:
    from tools.amarriner_gl_catalog import parse_gamescompleted_listing

    return parse_gamescompleted_listing(html)


def _load_sources(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in (".yaml", ".yml"):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _manual_days(yaml_days: Any) -> tuple[int | None, bool]:
    """Return (max_turn or None, is_latest). ``latest`` => full replay, cap in demos."""
    if isinstance(yaml_days, str) and yaml_days.strip().lower() == "latest":
        return (None, True)
    try:
        return (int(yaml_days), False)
    except (TypeError, ValueError):
        return (5, False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--map-pool", type=Path, default=ROOT / "data" / "gl_map_pool.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "human_openings" / "raw")
    ap.add_argument("--respect-cache", action="store_true", help="Skip download if games/<id>.zip exists")
    ap.add_argument("--sleep-s", type=float, default=0.75)
    ap.add_argument(
        "--save-listing-html",
        action="store_true",
        help="Write completed/*.html for the last page fetched per user query",
    )
    args = ap.parse_args()

    data = _load_sources(args.sources)
    manual: list[dict[str, Any]] = list(data.get("manual_games") or [])
    queries: list[dict[str, Any]] = list(data.get("completed_game_queries") or [])

    if not args.map_pool.is_file():
        print(f"[fetch] missing map pool: {args.map_pool}", file=sys.stderr)
        return 1
    std_ids = gl_std_map_ids(args.map_pool)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    games_dir = args.out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    completed_dir = args.out_dir / "completed"
    if args.save_listing_html:
        completed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    session = requests.Session()
    if not SECRETS.is_file():
        print(f"[fetch] missing {SECRETS} (needed for manual zip download)", file=sys.stderr)
        return 1
    creds = SECRETS.read_text(encoding="utf-8").strip().splitlines()
    if len(creds) < 2:
        print("[fetch] secrets.txt: line1 username, line2 password", file=sys.stderr)
        return 1
    user, pw = creds[0].strip(), creds[1].strip()
    if not _login(session, user, pw):
        print("[fetch] login failed", file=sys.stderr)
        return 1
    print("[fetch] logged in for replay zip downloads")

    ts = datetime.now(timezone.utc).isoformat()
    rows_out: list[dict[str, Any]] = []

    for mg in manual:
        gid = int(mg.get("game_id", 0) or 0)
        dmax, is_latest = _manual_days(mg.get("days", 5))
        dest = games_dir / f"{gid}.zip"
        if args.respect_cache and dest.is_file() and dest.stat().st_size > 100:
            status = "cached"
            err = None
        else:
            raw, err = _download_replay_zip(session, gid)
            if raw is None:
                status = "fail" if "in_progress" not in str(err) else "pending"
            else:
                dest.write_bytes(raw)
                status = "ok"
                err = None
        mid = mp0 = mp1 = None
        if status in ("ok", "cached") and dest.is_file():
            try:
                from tools.human_demo_rows import infer_training_meta_from_awbw_zip  # noqa: E402

                m = infer_training_meta_from_awbw_zip(dest, map_pool=args.map_pool)
                mid = m.get("map_id")
                mp0 = m.get("co0")
                mp1 = m.get("co1")
            except Exception:
                pass
        row = {
            "game_id": gid,
            "source": "manual",
            "requested_days": dmax,
            "latest_horizon": bool(is_latest),
            "fetch_status": status,
            "error": err,
            "zip_path": str(dest.relative_to(args.out_dir)) if status in ("ok", "cached") else None,
            "map_id": mid,
            "co_p0_id": mp0,
            "co_p1_id": mp1,
            "map_name": None,
            "ranked_standard_pool": None,
            "ts": ts,
        }
        rows_out.append(row)
        if err and status not in ("pending",):
            print(f"[fetch] manual games_id={gid} -> {err}", file=sys.stderr)
        elif status == "ok":
            print(f"[fetch] games_id={gid} saved {dest.name}")
        time.sleep(max(0.0, args.sleep_s))

    # --- Query GL completed listings for usernames (public HTML; no login) ---
    for q in queries:
        uname = str(q.get("username", "") or "").strip()
        limit = int(q.get("limit", 50) or 50)
        days = q.get("days", 5)
        try:
            req_days = int(days) if str(days).lower() != "latest" else None
        except (TypeError, ValueError):
            req_days = 5
        latest_q = str(days).lower() == "latest"
        if not uname:
            continue
        collected: list[dict[str, Any]] = []
        start = 1
        page = 0
        while len(collected) < limit and start < 50_000:
            if args.sleep_s > 0 and page > 0:
                time.sleep(args.sleep_s)
            html = _fetch_list_page(start)
            page += 1
            if args.save_listing_html:
                phtml = completed_dir / f"{uname}_page_{page}.html"
                phtml.write_text(html, encoding="utf-8")
            rows = _parse_gamescompleted_rows(html)
            if not rows:
                break
            low = uname.lower()
            for r in rows:
                mu = str(r.get("matchup") or "")
                if low not in mu.lower():
                    continue
                mid = r.get("map_id")
                if mid is None or int(mid) not in std_ids:
                    continue
                r2 = {**r, "source": f"user:{uname}"}
                collected.append(r2)
                if len(collected) >= limit:
                    break
            if len(rows) < 50:
                break
            start += 50

        for r in collected[:limit]:
            gid = int(r["games_id"])
            dest = games_dir / f"{gid}.zip"
            if args.respect_cache and dest.is_file() and dest.stat().st_size > 100:
                status = "cached"
                err = None
            else:
                raw, err = _download_replay_zip(session, gid)
                if raw is None:
                    status = "fail" if "in_progress" not in str(err or "") else "pending"
                else:
                    dest.write_bytes(raw)
                    status = "ok"
            row = {
                "game_id": gid,
                "source": "query",
                "source_username": uname,
                "requested_days": None if (latest_q or req_days is None) else int(req_days),
                "latest_horizon": bool(latest_q),
                "fetch_status": status,
                "error": err,
                "zip_path": str(dest.relative_to(args.out_dir)) if status in ("ok", "cached") else None,
                "map_id": r.get("map_id"),
                "map_name": r.get("map_name"),
                "co_p0_id": r.get("co_p0_id"),
                "co_p1_id": r.get("co_p1_id"),
                "matchup": r.get("matchup"),
                "ranked_standard_pool": True,
                "ts": ts,
            }
            rows_out.append(row)
            if status == "ok":
                print(f"[fetch] query {uname} games_id={gid} map={r.get('map_id')}")
            if err and status not in ("pending",):
                print(f"[fetch] games_id={gid} -> {err}", file=sys.stderr)
            time.sleep(max(0.0, args.sleep_s))

    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"[fetch] manifest rows={len(rows_out)} -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
