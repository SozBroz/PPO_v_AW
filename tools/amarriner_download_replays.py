#!/usr/bin/env python3
"""
Download AWBW replay ZIPs for games listed in ``data/amarriner_gl_std_catalog.json``.

Uses ``secrets.txt`` at repo root (line 1 username, line 2 password), same as
``tools/fetch_predeployed_units.py``. Writes ``{games_id}.zip`` under
``replays/amarriner_gl/`` by default.

By default, only games whose ``map_id`` is in the Global League **std** rotation
(``type == \"std\"`` in ``data/gl_map_pool.json``) are scheduled — same filter as
``tools/replay_state_diff.py``. Use ``--allow-non-gl-std-maps`` to include
completed games on maps no longer (or not yet) in that pool.

Games whose catalog row is missing ``co_p0_id`` or ``co_p1_id`` are **skipped**
(the engine cannot start without two CO ids). List them with
``python tools/amarriner_gl_catalog.py list-incomplete-cos``.

**In-progress games:** ``replay_download.php`` returns JSON ``Game is not over`` until
the match ends. Those rows are logged as ``kind: in_progress`` (manifest) and
``[download] PENDING`` (stdout); they do **not** increment ``fail`` or change the
process exit code when that is the only outcome.

Examples::

  python tools/amarriner_download_replays.py --map-id 123858 --dry-run
  python tools/amarriner_download_replays.py --map-id 123858 --sleep 1.0
  python tools/amarriner_download_replays.py --max-games 10

**Map colors (OS/BM):** after each successful download, this script runs
``tools.normalize_map_to_os_bm.run_normalize_map_to_os_bm`` for that game's ``map_id``
(unless ``--skip-os-bm-normalize``). That keeps ``data/maps/<id>.csv`` and
``p0_country_id`` aligned for oracle / desync_audit. Batch-fix existing maps with::

  python tools/normalize_map_to_os_bm.py --from-catalog
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.amarriner_catalog_cos import catalog_row_has_both_cos
from tools.gl_std_maps import gl_std_map_ids
from tools.normalize_map_to_os_bm import run_normalize_map_to_os_bm
from tools.oracle_zip_replay import replay_zip_has_action_stream

SECRETS = ROOT / "secrets.txt"
CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
MAP_POOL_DEFAULT = ROOT / "data" / "gl_map_pool.json"
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


def _replay_download_error_kind(body: bytes) -> Optional[str]:
    """
    If the mirror returns JSON instead of a zip (common for in-progress games),
    return a short machine-readable tag; otherwise None.
    """
    if not body or len(body) > 4096:
        return None
    stripped = body.lstrip()
    if not stripped.startswith(b"{"):
        return None
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not obj.get("err"):
        return None
    msg = str(obj.get("message") or "")
    low = msg.lower()
    if "not over" in low or "game is not over" in low:
        return "in_progress"
    return "json_err"


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
        kind = _replay_download_error_kind(r.content)
        if kind == "in_progress":
            return None, "in_progress: Game is not over (no replay zip until game ends)"
        if kind == "json_err":
            return None, f"mirror JSON error: {r.content[:400]!r}"
        return None, f"HTML body (likely login/error): {r.content[:200]!r}"
    if not r.content or len(r.content) < 100:
        kind = _replay_download_error_kind(r.content)
        if kind == "in_progress":
            return None, "in_progress: Game is not over (no replay zip until game ends)"
        if kind == "json_err":
            return None, f"mirror JSON error: {r.content[:400]!r}"
        return None, f"empty/small body ({len(r.content)} bytes)"
    if not zipfile.is_zipfile(io.BytesIO(r.content)):
        kind = _replay_download_error_kind(r.content)
        if kind == "in_progress":
            return None, "in_progress: Game is not over (no replay zip until game ends)"
        if kind == "json_err":
            return None, f"mirror JSON error: {r.content[:400]!r}"
        return None, "not a valid zip"
    return r.content, "ok"


def _load_catalog(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_map_after_download(catalog_row: dict[str, Any], map_pool: Path) -> None:
    """Remap map CSV to Orange Star / Blue Moon; best-effort (logs on failure)."""
    mid = catalog_row.get("map_id")
    if mid is None:
        return
    res = run_normalize_map_to_os_bm(
        int(mid),
        maps_dir=ROOT / "data" / "maps",
        pool_path=map_pool,
        dry_run=False,
        backup=True,
    )
    if res.ok:
        print(
            f"[os-bm] map_id={int(mid)} cells={res.changed_cells} "
            f"wrote_csv={res.wrote_csv} updated_pool={res.updated_pool}"
        )
    else:
        print(f"[os-bm] WARN map_id={mid}: {res.error}", file=sys.stderr)


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
    gl_std_map_ids_set: Optional[set[int]],
    allow_non_gl_std_maps: bool,
) -> list[dict[str, Any]]:
    games = data.get("games") or {}
    rows: list[dict[str, Any]] = []
    for _key, g in games.items():
        if not isinstance(g, dict):
            continue
        gid = int(g["games_id"])
        if games_ids is not None and gid not in games_ids:
            continue
        if not catalog_row_has_both_cos(g):
            continue
        mid = g.get("map_id")
        if (
            not allow_non_gl_std_maps
            and gl_std_map_ids_set is not None
            and mid is not None
            and int(mid) not in gl_std_map_ids_set
        ):
            continue
        if map_id is not None and (mid is None or int(mid) != map_id):
            continue
        if tier is not None and str(g.get("tier", "")) != tier:
            continue
        p0 = int(g["co_p0_id"])
        p1 = int(g["co_p1_id"])
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
    ap.add_argument(
        "--map-pool",
        type=Path,
        default=MAP_POOL_DEFAULT,
        help="Used with GL std map filter (default: data/gl_map_pool.json)",
    )
    ap.add_argument(
        "--allow-non-gl-std-maps",
        action="store_true",
        help="Download games whose map_id is not in the current Global League std rotation",
    )
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
    ap.add_argument(
        "--require-action-stream",
        action="store_true",
        help=(
            "After each download, require a gzip member with ``p:`` lines (ReplayVersion 2). "
            "If the zip is snapshot-only (ReplayVersion 1), delete the file and count as failure."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-os-bm-normalize",
        action="store_true",
        help=(
            "After a successful download, do not remap the map CSV to Orange Star/Blue Moon "
            "(default: normalize immediately so load_map matches GL seating)."
        ),
    )
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
    if not args.allow_non_gl_std_maps and not args.map_pool.is_file():
        print(f"[download] missing map pool: {args.map_pool}", file=sys.stderr)
        return 1

    std_ids: Optional[set[int]] = None
    if not args.allow_non_gl_std_maps:
        std_ids = gl_std_map_ids(args.map_pool)

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
        gl_std_map_ids_set=std_ids,
        allow_non_gl_std_maps=args.allow_non_gl_std_maps,
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

    ok = skip = fail = rv1 = awaiting_end = 0
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
                if err.startswith("in_progress:"):
                    awaiting_end += 1
                    rec = {"games_id": gid, "kind": "in_progress", "error": err, "ts": ts}
                    print(f"[download] PENDING games_id={gid} ({err})")
                else:
                    fail += 1
                    rec = {"games_id": gid, "kind": "error", "error": err, "ts": ts}
                    print(f"[download] FAIL games_id={gid} {err}")
                if manifest_f:
                    manifest_f.write(json.dumps(rec) + "\n")
                    manifest_f.flush()
            else:
                dest.write_bytes(raw)
                has_stream = replay_zip_has_action_stream(dest, games_id=gid)
                if not has_stream:
                    rv1 += 1
                    msg = (
                        "ReplayVersion 1 (no a<games_id> gzip with p: lines) — "
                        "oracle / desync_audit cannot replay; try re-download later or another mirror."
                    )
                    if args.require_action_stream:
                        try:
                            dest.unlink()
                        except OSError:
                            pass
                        fail += 1
                        rec = {"games_id": gid, "error": f"rv1_no_action_stream: {msg}", "ts": ts}
                        print(f"[download] FAIL games_id={gid} {msg}")
                        if manifest_f:
                            manifest_f.write(json.dumps(rec) + "\n")
                            manifest_f.flush()
                    else:
                        ok += 1
                        print(
                            f"[download] ok games_id={gid} -> {dest} ({len(raw)} bytes) "
                            f"[WARN: {msg}]"
                        )
                        if not args.skip_os_bm_normalize:
                            _normalize_map_after_download(g, args.map_pool)
                else:
                    ok += 1
                    print(f"[download] ok games_id={gid} -> {dest} ({len(raw)} bytes)")
                    if not args.skip_os_bm_normalize:
                        _normalize_map_after_download(g, args.map_pool)
            if i + 1 < len(planned) and args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        if manifest_f:
            manifest_f.close()

    extra = f" rv1_warn={rv1}" if rv1 and not args.require_action_stream else ""
    pend = f" awaiting_end={awaiting_end}" if awaiting_end else ""
    print(f"[download] done ok={ok} skip={skip} fail={fail}{pend}{extra}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
