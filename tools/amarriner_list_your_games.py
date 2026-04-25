#!/usr/bin/env python3
"""
List ``games_id`` values for the logged-in user from Amarriner ``yourgames.php``.

Credentials: repo root ``secrets.txt`` — line 1 username, line 2 password
(same as ``tools/amarriner_download_replays.py``).  Does not print passwords.

The HTML is scraped for links to ``game.php`` with a numeric id (your active
or waiting games).  Use with :func:`tools.amarriner_live_meta.resolve_games_meta`
to fill ``map_id`` / COs for :func:`tools.desync_audit_amarriner_live.build_live_engine_state`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SECRETS = ROOT / "secrets.txt"
BASE_URL = "https://awbw.amarriner.com"
LOGIN_URL = f"{BASE_URL}/login.php"
# Default browser listing for “your games” uses yourTurn=0 (all rows / same as site filter).
# See https://awbw.amarriner.com/yourgames.php?yourTurn=0
YOURGAMES_URL = f"{BASE_URL}/yourgames.php"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


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


def find_game_ids_in_html(html: str) -> list[int]:
    """
    Best-effort extraction of Amarriner game ids from a page.
    Tries game.php?...=N and standalone games_id= in links and scripts.
    """
    ids: set[int] = set()
    for m in re.finditer(
        r"game\.php\?[^\"'>\s]+",
        html,
        flags=re.IGNORECASE,
    ):
        for sub in re.finditer(r"(?:^|[?&])(?:games_id|id|gameId)=(\d+)", m.group(0), re.I):
            ids.add(int(sub.group(1)))
    for m in re.finditer(
        r"(?:^|[?&])games_id=(\d+)",
        html,
        flags=re.IGNORECASE,
    ):
        ids.add(int(m.group(1)))
    return sorted(ids)


def list_your_games_ids(
    session: requests.Session,
    *,
    timeout_s: float = 25.0,
    your_turn: int | None = 0,
) -> list[int]:
    """
    :param your_turn: If an ``int``, request ``yourgames.php?yourTurn=<int>`` (default ``0``).
        If ``None``, request ``yourgames.php`` with no query string (legacy).
    """
    params = None if your_turn is None else {"yourTurn": int(your_turn)}
    r = session.get(
        YOURGAMES_URL,
        params=params,
        headers=HEADERS,
        timeout=timeout_s,
        allow_redirects=True,
    )
    r.raise_for_status()
    return find_game_ids_in_html(r.text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object {\"games_id\": [ ... ]} on stdout",
    )
    ap.add_argument(
        "--your-turn",
        type=int,
        default=0,
        help="yourgames.php query yourTurn= (default 0). Matches the site listing URL.",
    )
    ap.add_argument(
        "--no-your-turn-param",
        action="store_true",
        help="Request plain yourgames.php with no yourTurn= filter (legacy).",
    )
    args = ap.parse_args()

    if not SECRETS.is_file():
        print(f"[yourgames] missing {SECRETS}", file=sys.stderr)
        return 1
    lines = [ln.strip() for ln in SECRETS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        print(
            f"[yourgames] {SECRETS} needs line 1=username, line 2=password",
            file=sys.stderr,
        )
        return 1
    user, password = lines[0], lines[1]

    sess = requests.Session()
    if not _login(sess, user, password):
        print("[yourgames] login failed", file=sys.stderr)
        return 1
    yt = None if args.no_your_turn_param else int(args.your_turn)
    gids = list_your_games_ids(sess, your_turn=yt)
    if args.json:
        print(json.dumps({"games_id": gids}, indent=2))
    else:
        for gid in gids:
            print(gid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
