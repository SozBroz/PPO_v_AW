"""
AWBW authenticated scraper — fetches Global League map pool + tiers.

Endpoint: POST https://awbw.amarriner.com/api/league/maps.php
          Body: {"method": "listMaps"}

Usage:
    python data/fetch_awbw.py

Reads credentials from secrets.txt (line 1 = username, line 2 = password).
Outputs:
    data/gl_maps.json      — full raw API response
    data/gl_map_pool.json  — clean map pool: [{map_id, name, type, tiers, unit_bans, ...}]
"""

import json
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
SECRETS = ROOT / "secrets.txt"
OUT_RAW = ROOT / "data" / "gl_maps_raw.json"
OUT_CLEAN = ROOT / "data" / "gl_map_pool.json"

BASE_URL = "https://awbw.amarriner.com"
LOGIN_URL = f"{BASE_URL}/login.php"
API_URL = f"{BASE_URL}/api/league/maps.php"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": f"{BASE_URL}/newleaguesetups.php",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
}


def log(msg: str):
    print(msg, flush=True)


def load_credentials() -> tuple[str, str]:
    lines = SECRETS.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        sys.exit("secrets.txt must have username on line 1, password on line 2")
    return lines[0].strip(), lines[1].strip()


def login(session: requests.Session, username: str, password: str) -> bool:
    """POST credentials to AWBW and verify authentication."""
    r = session.get(LOGIN_URL, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=15)
    r.raise_for_status()

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

    action = BASE_URL + "/login.php"
    if form:
        raw_action = form.get("action", "login.php")
        action = raw_action if raw_action.startswith("http") else f"{BASE_URL}/{raw_action.lstrip('/')}"

    log(f"[login] POST {action} user={username}")
    r2 = session.post(
        action,
        data=payload,
        headers={"User-Agent": HEADERS["User-Agent"]},
        timeout=15,
        allow_redirects=True,
    )
    r2.raise_for_status()

    authed = "logout" in r2.text.lower() or username.lower() in r2.text.lower()
    if authed:
        log("[login] Authenticated successfully")
    else:
        log("[login] WARNING: authentication uncertain — proceeding anyway")
    return authed


def fetch_map_pool(session: requests.Session) -> dict:
    """Call the Vue app's listMaps endpoint."""
    log(f"[api] POST {API_URL} method=listMaps")
    r = session.post(
        API_URL,
        json={"method": "listMaps"},
        headers=HEADERS,
        timeout=20,
    )
    log(f"[api] Response: {r.status_code} {r.headers.get('content-type', '')}")
    r.raise_for_status()

    data = r.json()
    return data


def clean_map_pool(raw: dict) -> list[dict]:
    """
    Extract essential map pool info from the listMaps API response.

    Raw structure:
      maps[].id, name, type, tiers[], unitBans[], unitLimit, capLimit
      tiers[].name, enabled, cos[].id  (cos keyed only by id)
      cos{}  — top-level CO lookup: {"1": {"name": "Andy", ...}, ...}
    """
    co_lookup: dict[str, str] = {
        k: v["name"] for k, v in raw.get("cos", {}).items()
    }

    cleaned = []
    for m in raw.get("maps", []):
        tiers = []
        for tier in m.get("tiers", []):
            co_ids = [co["id"] for co in tier.get("cos", [])]
            tiers.append({
                "tier_name": tier.get("name", ""),
                "enabled": tier.get("enabled", True),
                "co_ids": co_ids,
                "co_names": [co_lookup.get(str(cid), f"CO#{cid}") for cid in co_ids],
            })
        cleaned.append({
            "map_id": m.get("id"),
            "name": m.get("name", ""),
            "type": m.get("type", ""),       # "std" | "hf" | "fog"
            "unit_bans": m.get("unitBans", []),
            "unit_limit": m.get("unitLimit", 0),
            "cap_limit": m.get("capLimit", 0),
            "tiers": tiers,
        })
    return cleaned


MAP_CSV_URL = f"{BASE_URL}/text_map.php"
MAPS_DIR = ROOT / "data" / "maps"


def download_map_csvs(session: requests.Session, maps: list[dict], *, force: bool = False) -> None:
    """
    Download terrain CSV for each map into data/maps/<map_id>.csv.

    AWBW exports terrain only (no predeployed units). Optional
    data/maps/<map_id>_units.json is maintained separately — see engine/predeployed.py.

    Parameters
    ----------
    force:  If True, delete existing CSVs before downloading so every map is
            re-fetched from AWBW (useful after terrain ID mapping fixes).
    """
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    skipped = 0
    failed = []

    for m in maps:
        map_id = m["map_id"]
        dest = MAPS_DIR / f"{map_id}.csv"

        if dest.exists():
            if force:
                dest.unlink()
                log(f"  [del]  {map_id} — removed cached CSV (force re-download)")
            else:
                log(f"  [skip] {map_id} already on disk")
                skipped += 1
                continue

        success = False
        for attempt in range(3):
            try:
                r = session.get(
                    MAP_CSV_URL,
                    params={"maps_id": map_id, "download": "csv"},
                    headers={"User-Agent": HEADERS["User-Agent"]},
                    timeout=15,
                )
                if r.status_code == 503:
                    wait = 10 * (attempt + 1)
                    log(f"  [503]  {map_id} — server busy, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                dest.write_bytes(r.content)
                log(f"  [dl]   {map_id} -> {dest.name}  ({len(r.content)} bytes)")
                ok += 1
                success = True
                break
            except requests.HTTPError as e:
                log(f"  [FAIL] {map_id}: {e}")
                break
            except Exception as e:
                log(f"  [FAIL] {map_id}: {e}")
                break

        if not success and map_id not in failed:
            failed.append(map_id)

        time.sleep(0.3)  # be a polite guest

    log(f"\n[maps] {ok} downloaded, {skipped} already cached, {len(failed)} failed")
    if failed:
        log(f"[maps] Failed IDs: {failed}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch AWBW Global League map pool and terrain CSVs.")
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download all map CSVs even if they already exist on disk.",
    )
    args = parser.parse_args()

    username, password = load_credentials()
    session = requests.Session()

    login(session, username, password)

    raw = fetch_map_pool(session)

    OUT_RAW.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[write] Raw response -> {OUT_RAW}")

    maps = clean_map_pool(raw)
    OUT_CLEAN.write_text(json.dumps(maps, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[write] Clean map pool -> {OUT_CLEAN}")

    log(f"\n[pool] {len(maps)} maps")
    for m in maps:
        tiers_summary = ", ".join(t["tier_name"] for t in m["tiers"])
        bans = ", ".join(m["unit_bans"]) if m["unit_bans"] else "none"
        log(f"  [{m['type']:3s}] {m['map_id']:6} | {m['name'][:40]:<40} | tiers: {tiers_summary} | bans: {bans}")

    log(f"\n[maps] Downloading terrain CSVs {'(FORCE mode — re-fetching all)' if args.force else ''}...")
    download_map_csvs(session, maps, force=args.force)


if __name__ == "__main__":
    main()
