#!/usr/bin/env python3
"""
Download AWBW replay ZIPs for games listed in ``data/amarriner_gl_std_catalog.json``.

Uses ``secrets.txt`` at repo root (line 1 username, line 2 password), same as
``tools/fetch_predeployed_units.py``. Writes ``{games_id}.zip`` under
``replays/amarriner_gl/`` by default.

Examples::

  python tools/amarriner_download_replays.py --map-id 123858 --dry-run
  python tools/amarriner_download_replays.py --map-id 123858 --sleep 1.0
  python tools/amarriner_download_replays.py --max-games 10
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT / "secrets.txt"
CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
OUT_DIR_DEFAULT = ROOT / "replays" / "amarriner_gl"
BASE_URL = "https://awbw.amarriner.com"
LOGIN_URL = f"{BASE_URL}/login.php"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


def _login(session: requests.Session, username: str, password: str) -> bool:
    r = session.get(LOGIN_URL, headers=HEADERS, timeout=15)
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
        LOGIN_URL, data=payload, headers=HEADERS, timeout=15, allow_redirects=True
    )
    return "logout" in r2.text.lower() or username.lower() in r2.text.lower()


def _download_replay_zip(session: requests.Session, game_id: int) -> tuple[Optional[bytes], str]:
    url = f"{BASE_URL}/replay_download.php?games_id={game_id}"
    try:
        r = session.get(url, headers=HEADERS, timeout=60)
    except requests.RequestException as e:
        return None, f"network: {e}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "html" in ctype and len(r.content) < 8000:
        return None, f"HTML body (likely login/error): {r.content[:200]!r}"
    if not r.content or len(r.content) < 100:
        return None, f"empty/small body ({len(r.content)} bytes)"
    if not zipfile.is_zipfile(io.BytesIO(r.content)):
        return None, "not a valid zip"
    return r.content, "ok"


def _load_catalog(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_games(
    data: dict[str, Any],
    *,
    map_id: Optional[int],
    co_p0_id: Optional[int],
    co_p1_id: Optional[int],
    tier: Optional[str],
    mirror_andy: bool,
    games_ids: Optional[set[int]],
    max_games: Optional[int],
) -> list[dict[str, Any]]:
    games = data.get("games") or {}
    rows: list[dict[str, Any]] = []
    for _key, g in games.items():
        if not isinstance(g, dict):
            continue
        gid = int(g["games_id"])
        if games_ids is not None and gid not in games_ids:
            continue
        if map_id is not None and int(g.get("map_id", -1)) != map_id:
            continue
        if tier is not None and str(g.get("tier", "")) != tier:
            continue
        p0 = int(g.get("co_p0_id", -1))
        p1 = int(g.get("co_p1_id", -1))
        if mirror_andy and (p0 != 1 or p1 != 1):
            continue
        if co_p0_id is not None and p0 != co_p0_id:
            continue
        if co_p1_id is not None and p1 != co_p1_id:
            continue
        rows.append(g)
    rows.sort(key=lambda x: int(x["games_id"]))
    if max_games is not None:
        rows = rows[: max(0, max_games)]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--map-id", type=int, default=None)
    ap.add_argument("--co-p0-id", type=int, default=None)
    ap.add_argument("--co-p1-id", type=int, default=None)
    ap.add_argument("--tier", type=str, default=None)
    ap.add_argument(
        "--mirror-andy",
        action="store_true",
        help="Only games with CO Andy (id 1) on both sides",
    )
    ap.add_argument("--games-id", type=int, action="append", default=None)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.75, help="Seconds between downloads")
    ap.add_argument("--force", action="store_true", help="Overwrite existing zip")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Append JSON lines for failures (games_id, error, ts)",
    )
    args = ap.parse_args()

    if not args.catalog.is_file():
        print(f"[download] missing catalog: {args.catalog}", file=sys.stderr)
        return 1

    gid_set = set(args.games_id) if args.games_id else None
    data = _load_catalog(args.catalog)
    planned = _iter_games(
        data,
        map_id=args.map_id,
        co_p0_id=args.co_p0_id,
        co_p1_id=args.co_p1_id,
        tier=args.tier,
        mirror_andy=args.mirror_andy,
        games_ids=gid_set,
        max_games=args.max_games,
    )
    print(f"[download] planned games: {len(planned)}")

    if args.dry_run:
        for g in planned[:20]:
            print(f"  would fetch games_id={g['games_id']} map={g.get('map_id')} tier={g.get('tier')}")
        if len(planned) > 20:
            print(f"  ... and {len(planned) - 20} more")
        return 0

    if not SECRETS.is_file():
        print(f"[download] missing {SECRETS}", file=sys.stderr)
        return 1
    creds = SECRETS.read_text(encoding="utf-8").strip().splitlines()
    if len(creds) < 2:
        print("[download] secrets.txt needs username line 1, password line 2", file=sys.stderr)
        return 1
    user, pw = creds[0].strip(), creds[1].strip()

    session = requests.Session()
    if not _login(session, user, pw):
        print("[download] login failed", file=sys.stderr)
        return 1
    print("[download] logged in")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_f = None
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_f = open(args.manifest, "a", encoding="utf-8")

    ok = skip = fail = 0
    ts = datetime.now(timezone.utc).isoformat()
    try:
        for i, g in enumerate(planned):
            gid = int(g["games_id"])
            dest = args.out_dir / f"{gid}.zip"
            if dest.is_file() and dest.stat().st_size > 0 and not args.force:
                skip += 1
                continue
            raw, err = _download_replay_zip(session, gid)
            if raw is None:
                fail += 1
                rec = {"games_id": gid, "error": err, "ts": ts}
                print(f"[download] FAIL games_id={gid} {err}")
                if manifest_f:
                    manifest_f.write(json.dumps(rec) + "\n")
                    manifest_f.flush()
            else:
                dest.write_bytes(raw)
                ok += 1
                print(f"[download] ok games_id={gid} -> {dest} ({len(raw)} bytes)")
            if i + 1 < len(planned) and args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        if manifest_f:
            manifest_f.close()

    print(f"[download] done ok={ok} skip={skip} fail={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
