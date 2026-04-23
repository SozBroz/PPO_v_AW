"""
Ingest predeployed unit positions for all maps in gl_map_pool.json.

Strategy
--------
For each map_id, download a completed game's replay ZIP via:
  GET https://awbw.amarriner.com/replay_download.php?games_id=<game_id>

The replay ZIP contains one or more gzip-compressed PHP-serialized snapshots.
The very first line (turn 0 = initial game state) holds every awbwUnit with:
  units_x  (0-based col), units_y (0-based row), units_name, units_players_id

We also fetch the game page to get playersInfo so we can map players_id → player 0/1.

Output: data/maps/<map_id>_units.json (schema_version 1, compatible with engine/predeployed.py)

Usage
-----
  python tools/fetch_predeployed_units.py                 # process full pool
  python tools/fetch_predeployed_units.py --type std      # standard maps only
  python tools/fetch_predeployed_units.py --map-id 180298 # single map
  python tools/fetch_predeployed_units.py --dry-run       # print only, no writes
  python tools/fetch_predeployed_units.py --force         # overwrite existing files
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
POOL_PATH  = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR   = ROOT / "data" / "maps"
SECRETS    = ROOT / "secrets.txt"
BASE_URL   = "https://awbw.amarriner.com"
LOGIN_URL  = f"{BASE_URL}/login.php"
HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# ---------------------------------------------------------------------------
# AWBW unit name → engine UnitType enum name.
# Phase 11Z: derived from the canon in ``engine/unit_naming.py``. Kept as a
# public dict so any external/out-of-tree caller still sees the same
# ``AWBW_NAME_TO_UNIT_TYPE`` symbol. To add a spelling, edit
# ``engine/unit_naming.py`` and not this dict.
# ---------------------------------------------------------------------------
def _build_awbw_name_to_unit_type() -> dict[str, str]:
    # Local import to avoid forcing the engine import at module load time
    # for callers that only want the http helpers.
    from engine.unit_naming import all_known_aliases

    return {alias: ut.name for alias, ut in all_known_aliases().items()}


AWBW_NAME_TO_UNIT_TYPE: dict[str, str] = _build_awbw_name_to_unit_type()


# ---------------------------------------------------------------------------
# HTTP / auth helpers
# ---------------------------------------------------------------------------
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
    authed = "logout" in r2.text.lower() or username.lower() in r2.text.lower()
    return authed


def _extract_js_var(html: str, name: str) -> Optional[str]:
    """Extract the value of a JS variable (var/let/const) using bracket counting."""
    m = re.search(rf'(?:var|let|const)\s+{re.escape(name)}\s*=\s*', html)
    if not m:
        return None
    start = m.end()
    if start >= len(html):
        return None
    first = html[start]
    if first in ('{', '['):
        close = '}' if first == '{' else ']'
        depth = 0
        i = start
        in_str = False
        str_char = ''
        while i < len(html):
            c = html[i]
            if in_str:
                if c == '\\':
                    i += 2
                    continue
                if c == str_char:
                    in_str = False
            else:
                if c in ('"', "'"):
                    in_str = True
                    str_char = c
                elif c == first:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        return html[start:i + 1]
            i += 1
    else:
        end = html.find(';', start)
        return html[start:end] if end > start else None
    return None


# ---------------------------------------------------------------------------
# Replay download and parsing
# ---------------------------------------------------------------------------
def _download_replay_zip(session: requests.Session, game_id: int) -> Optional[bytes]:
    """Download the replay ZIP for a game. Returns raw bytes or None on failure."""
    url = f"{BASE_URL}/replay_download.php?games_id={game_id}"
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        print(f"    [network error] {e}")
        return None
    if r.status_code != 200:
        print(f"    [HTTP {r.status_code}] {url}")
        return None
    ctype = r.headers.get("Content-Type", "")
    if "html" in ctype and len(r.content) < 1000:
        # Error response
        print(f"    [error response] {r.content[:100]!r}")
        return None
    return r.content


def _parse_replay_zip(zip_bytes: bytes, game_id: int) -> tuple[Optional[str], Optional[str]]:
    """
    Extract (turn0_snapshot_line, action_stream_first_line) from the replay ZIP.

    Returns a tuple of (snapshot_line0, action_line0), either may be None on error.
    """
    snap_line: Optional[str]   = None
    action_line: Optional[str] = None
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            snap_entry   = str(game_id)
            action_entry = f"a{game_id}"
            if snap_entry not in names:
                candidates = [n for n in names if not n.startswith("a")]
                snap_entry = candidates[0] if candidates else None
            if action_entry not in names:
                candidates = [n for n in names if n.startswith("a")]
                action_entry = candidates[0] if candidates else None

            for entry, target in [(snap_entry, "snap"), (action_entry, "action")]:
                if not entry:
                    continue
                try:
                    raw = zf.read(entry)
                    with gzip.open(io.BytesIO(raw)) as gz:
                        text = gz.read().decode("utf-8", errors="replace")
                    lines = text.split("\n")
                    if target == "snap":
                        snap_line = lines[0] if lines else None
                    else:
                        action_line = lines[0] if lines else None
                except Exception as e:
                    print(f"    [entry {entry!r} error] {e}")
    except Exception as e:
        print(f"    [zip parse error] {e}")
    return snap_line, action_line


def _parse_turn0_snapshot(zip_bytes: bytes, game_id: int) -> Optional[str]:
    """Backwards-compat wrapper: return just the turn-0 snapshot line."""
    snap, _ = _parse_replay_zip(zip_bytes, game_id)
    return snap


def _read_php_field(chunk: str, field: str) -> Optional[str]:
    """Read a single PHP field value from an object chunk. Returns raw value string or None."""
    fp = f's:{len(field.encode())}:"{field}";'
    fi = chunk.find(fp)
    if fi < 0:
        return None
    rest = chunk[fi + len(fp):]
    return rest[:60].split(";")[0]


def _extract_awbw_objects(line: str, cls: str) -> list[dict]:
    """
    Extract all PHP-serialized objects of class `cls` from a snapshot line.

    Official AWBW replay format uses short field names (no 'units_' prefix):
      id, games_id, players_id, name, x, y, hit_points, fuel, ammo, ...
    """
    pat = re.compile(rf'O:{len(cls)}:"{re.escape(cls)}":\d+:\{{')
    results = []
    for m in pat.finditer(line):
        chunk = line[m.end():]
        obj: dict = {}
        # Short field names used in official AWBW replays
        int_fields  = ["id", "games_id", "players_id", "x", "y", "hit_points",
                        "fuel", "ammo", "movement_points"]
        str_fields  = ["name"]
        for field in int_fields:
            raw = _read_php_field(chunk, field)
            if raw and (raw.startswith("i:") or raw.startswith("d:")):
                try:
                    obj[field] = int(float(raw[2:]))
                except ValueError:
                    pass
        for field in str_fields:
            raw = _read_php_field(chunk, field)
            if raw and raw.startswith("s:"):
                m2 = re.search(r'"([^"]*)"', raw)
                if m2:
                    obj[field] = m2.group(1)
        if "x" in obj and "y" in obj:
            results.append(obj)
    return results


def _extract_player_order_from_actions(action_line: str) -> dict[int, int]:
    """
    Derive player turn order from the action stream.

    Each action-stream line starts with  p:<players_id>;d:<day>;...
    Iterating lines in order gives turns: p0 on day 1, p1 on day 1, p0 on day 2, ...

    We read the `p:` prefix from each line and assign player-index 0, 1, 0, 1, ...
    as they first appear (deduplication in first-seen order).

    Returns {players_id: 0 or 1}.
    """
    pid_to_player: dict[int, int] = {}
    # action_line is line 0 of the action stream — that's the first turn, player 0
    m = re.match(r'p:(\d+);', action_line)
    if m:
        pid_to_player[int(m.group(1))] = 0
    return pid_to_player


def _extract_player_order_from_snapshot(snapshot_line: str) -> dict[int, int]:
    """
    Fallback: extract all players_id from awbwUnit objects in the snapshot,
    then determine order by finding who appears first in the action stream or
    by reading awbwGame's player list order.

    AWBW stores players in order inside the awbwGame object.
    Look for the awbwGame's "players" array: a:<N>:{i:0;O:10:"awbwPlayer":...i:1;O:10:...}
    and read player IDs in array order (index 0 = first player).
    """
    pid_to_player: dict[int, int] = {}
    # Find the players array inside awbwGame
    # awbwGame contains:  s:7:"players";a:2:{i:0;O:10:"awbwPlayer":...i:1;O:10:"awbwPlayer"...}
    game_pat = re.compile(r's:7:"players";a:\d+:\{(.*?)s:9:"buildings"', re.DOTALL)
    gm = game_pat.search(snapshot_line)
    if not gm:
        return pid_to_player
    players_block = gm.group(1)
    # Find all awbwPlayer objects in array order
    player_pat = re.compile(r'O:\d+:"awbwPlayer":\d+:\{')
    idx = 0
    for pm in player_pat.finditer(players_block):
        chunk = players_block[pm.end():]
        # Read "id" field
        raw = _read_php_field(chunk, "id")
        if raw and raw.startswith("i:"):
            try:
                pid = int(raw[2:])
                pid_to_player[pid] = idx
                idx += 1
            except ValueError:
                pass
        if idx >= 2:
            break
    return pid_to_player


def _get_game_player_order(
    session: requests.Session,
    game_id: int,
) -> dict[int, int]:
    """
    Fallback: fetch game.php and parse playersInfo to get player order mapping.
    Returns {players_id: 0 or 1}.
    """
    try:
        r = session.get(
            f"{BASE_URL}/game.php?games_id={game_id}",
            headers=HEADERS,
            timeout=15,
        )
    except requests.RequestException:
        return {}
    raw = _extract_js_var(r.text, "playersInfo")
    if not raw:
        return {}
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    result: dict[int, int] = {}
    for pid_str, pdata in info.items():
        order = pdata.get("players_order", 1)
        result[int(pid_str)] = order - 1
    return result


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------
def _get_completed_game_ids(
    session: requests.Session, map_id: int, limit: int = 5
) -> list[int]:
    """Return up to `limit` completed game IDs for a given map."""
    url = f"{BASE_URL}/gamescompleted.php?maps_id={map_id}"
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    game_ids: list[int] = []
    seen: set[int] = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r'games_id=(\d+)(?:&|$)', a["href"])
        if m and "game.php" in a["href"]:
            gid = int(m.group(1))
            if gid not in seen:
                seen.add(gid)
                game_ids.append(gid)
            if len(game_ids) >= limit:
                break
    return game_ids


def fetch_predeploys_for_map(
    session: requests.Session,
    map_id: int,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """
    Fetch and write predeploy units for one map.
    Returns True on success, False on skip/failure.
    """
    out_path = MAPS_DIR / f"{map_id}_units.json"
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to overwrite)")
        return True

    map_csv = MAPS_DIR / f"{map_id}.csv"
    if not map_csv.exists():
        print(f"  [skip] no map CSV at {map_csv} — cannot validate bounds")
        return False

    # Load map dimensions from CSV (height = row count, width = first row col count)
    rows = [line.strip() for line in map_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    map_height = len(rows)
    map_width  = len(rows[0].split(",")) if rows else 0

    # Get completed games
    game_ids = _get_completed_game_ids(session, map_id)
    if not game_ids:
        print(f"  [skip] no completed games found for map {map_id}")
        return False
    print(f"  Found {len(game_ids)} completed game(s): {game_ids[:5]}")

    # Try each game until we get a valid turn-0 snapshot with units
    units_list: Optional[list[dict]] = None
    pid_to_player: dict[int, int] = {}
    used_game_id: Optional[int] = None

    for game_id in game_ids:
        print(f"  Trying game {game_id}...")
        zip_bytes = _download_replay_zip(session, game_id)
        if not zip_bytes:
            time.sleep(0.5)
            continue

        snap_line, action_line = _parse_replay_zip(zip_bytes, game_id)
        if not snap_line:
            time.sleep(0.5)
            continue

        units = _extract_awbw_objects(snap_line, "awbwUnit")
        print(f"    turn-0 awbwUnit count: {len(units)}")
        if len(units) == 0:
            # Try next game before concluding no predeploys
            time.sleep(0.5)
            continue

        # Determine player order: action stream is authoritative
        pid_map: dict[int, int] = {}
        if action_line:
            pid_map = _extract_player_order_from_actions(action_line)
        if not pid_map:
            pid_map = _extract_player_order_from_snapshot(snap_line)
        if not pid_map:
            print(f"    No p0 from actions, fetching game.php...")
            time.sleep(0.5)
            pid_map = _get_game_player_order(session, game_id)

        # We only know player 0 from the first action line.
        # Infer player 1: any players_id in units not in pid_map = player 1.
        all_pids = {u.get("players_id") for u in units if u.get("players_id") is not None}
        p0_pids  = set(pid_map.keys())
        p1_pids  = all_pids - p0_pids
        for p1_pid in p1_pids:
            pid_map[p1_pid] = 1

        if not pid_map:
            print(f"    Cannot determine player order for game {game_id}, skipping")
            time.sleep(0.5)
            continue

        units_list = units
        pid_to_player = pid_map
        used_game_id = game_id
        break

    if units_list is None:
        # All games had units=0 — accept that as "map has no predeploys"
        for game_id in game_ids[:1]:
            zip_bytes = _download_replay_zip(session, game_id)
            if not zip_bytes:
                continue
            snap_line, action_line = _parse_replay_zip(zip_bytes, game_id)
            if snap_line:
                units_list = []
                pid_to_player = {}
                used_game_id = game_id
                print(f"    Accepted 0-unit result from game {game_id} (map has no predeploys)")
                break

    if units_list is None:
        print(f"  [fail] could not extract valid snapshot for map {map_id}")
        return False

    # Build output records
    records: list[dict] = []
    skipped = 0
    for u in units_list:
        raw_name = u.get("name", "")
        unit_type = AWBW_NAME_TO_UNIT_TYPE.get(raw_name)
        if not unit_type:
            print(f"    [warn] unknown unit name {raw_name!r}, skipping")
            skipped += 1
            continue

        pid = u.get("players_id")
        player = pid_to_player.get(pid)
        if player is None:
            print(f"    [warn] unknown players_id {pid}, skipping unit")
            skipped += 1
            continue

        col = int(u["x"])   # AWBW 0-based column
        row = int(u["y"])   # AWBW 0-based row
        hp_awbw = u.get("hit_points", 10)
        # AWBW uses 1-10 HP; engine uses 1-100
        hp_engine = min(100, max(1, int(round(hp_awbw * 10))))

        # Bounds check
        if map_width and map_height:
            if not (0 <= col < map_width and 0 <= row < map_height):
                print(f"    [warn] unit {raw_name} at ({col},{row}) out of bounds "
                      f"({map_width}x{map_height}), skipping")
                skipped += 1
                continue

        rec: dict = {
            "row":       row,
            "col":       col,
            "player":    player,
            "unit_type": unit_type,
        }
        if hp_engine < 100:
            rec["hp"] = hp_engine
        records.append(rec)

    print(f"  Extracted {len(records)} units ({skipped} skipped) from game {used_game_id}")

    output = {
        "schema_version": 1,
        "_source": f"fetched from awbw game {used_game_id} turn-0 snapshot",
        "units": records,
    }

    if dry_run:
        print(f"  [dry-run] would write {out_path.name}:")
        print(json.dumps(output, indent=2)[:500])
        return True

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AWBW predeploy units for GL map pool")
    parser.add_argument("--map-id",  type=int, help="Process a single map ID")
    parser.add_argument("--type",    help="Filter by map type (e.g. 'std')")
    parser.add_argument("--dry-run", action="store_true", help="Print output, do not write files")
    parser.add_argument("--force",   action="store_true", help="Overwrite existing _units.json files")
    args = parser.parse_args()

    # Auth
    if not SECRETS.exists():
        sys.exit(f"secrets.txt not found at {SECRETS}")
    creds = SECRETS.read_text(encoding="utf-8").strip().splitlines()
    if len(creds) < 2:
        sys.exit("secrets.txt must have username on line 1, password on line 2")
    username, password = creds[0].strip(), creds[1].strip()

    session = requests.Session()
    authed = _login(session, username, password)
    print(f"[login] authenticated={authed}")

    # Load pool
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    seen_ids: set[int] = set()
    maps_to_process: list[dict] = []
    for entry in pool:
        mid = entry.get("map_id")
        if mid is None or mid in seen_ids:
            continue
        seen_ids.add(mid)
        if args.map_id and mid != args.map_id:
            continue
        if args.type and entry.get("type") != args.type:
            continue
        maps_to_process.append(entry)

    print(f"Processing {len(maps_to_process)} map(s)...\n")

    ok = err = skip = 0
    for entry in maps_to_process:
        mid = entry["map_id"]
        name = entry.get("name", "?")
        print(f"\n--- map {mid}: {name} ---")
        result = fetch_predeploys_for_map(
            session, mid, dry_run=args.dry_run, force=args.force
        )
        if result:
            ok += 1
        else:
            err += 1
        time.sleep(1.0)  # polite delay between maps

    print(f"\nDone: {ok} succeeded, {err} failed/skipped")


if __name__ == "__main__":
    main()
