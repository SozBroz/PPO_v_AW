"""
Resolve :class:`dict` metadata for a ``games_id`` from on-disk Amarriner catalog JSONs
or from a **game page** HTML ``game.php?games_id=…`` (``maps_id`` link + CO portraits).

Searches (in order) ``amarriner_gl_std_catalog.json``,
``amarriner_gl_extras_catalog.json``, ``amarriner_gl_current_list_250.json`` under
``<repo>/data/``.  Returns the first match with ``map_id`` and both CO ids, or
``None``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools.amarriner_catalog_cos import catalog_row_has_both_cos

_CATALOG_CANDIDATES = (
    "amarriner_gl_std_catalog.json",
    "amarriner_gl_extras_catalog.json",
    "amarriner_gl_current_list_250.json",
)


def resolve_games_meta(
    games_id: int, *, repo_root: Path | None = None
) -> dict[str, Any] | None:
    """
    Return a catalog row for ``games_id`` (includes ``map_id``, ``co_p*``, ``tier``)
    or ``None`` if no file contains that game with complete CO data.
    """
    root = repo_root
    if root is None:
        root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    for name in _CATALOG_CANDIDATES:
        path = data_dir / name
        if not path.is_file():
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        games = blob.get("games")
        if not isinstance(games, dict):
            continue
        key = str(int(games_id))
        row = games.get(key)
        if not isinstance(row, dict):
            continue
        if not catalog_row_has_both_cos(row):
            continue
        if int(row.get("games_id", -1)) != int(games_id):
            continue
        return row
    return None


def infer_meta_from_game_page_html(
    html: str, games_id: int, *, default_tier: str = "T2"
) -> dict[str, Any] | None:
    """
    Best-effort ``games_id``/``map_id``/``co_p*`` from Amarriner ``game.php`` HTML
    (not every layout exposes both COs; returns ``None`` if insufficient).
    """
    m = re.search(r"prevmaps\.php\?maps_id=(\d+)", html, re.I)
    if not m:
        m = re.search(r"maps_id['\"]?\s*[:=]\s*(\d+)", html, re.I)
    if not m:
        return None
    map_id = int(m.group(1))
    from tools.amarriner_gl_catalog import (  # lazy: same CO portrait regex as GL listings
        _RE_PLAYER_CO,
    )

    co_ids = [int(x) for x in _RE_PLAYER_CO.findall(html)]
    if len(co_ids) < 2:
        return None
    return {
        "games_id": int(games_id),
        "map_id": map_id,
        "co_p0_id": int(co_ids[0]),
        "co_p1_id": int(co_ids[1]),
        "tier": str(default_tier),
        "matchup": "",
    }


def meta_from_first_snap_and_map_id(
    first_snap: dict[str, Any], map_id: int, games_id: int, *, tier: str = "T2"
) -> dict[str, Any] | None:
    """
    Build catalog-shaped meta for live replay using the same **player order** rule
    as :func:`tools.oracle_zip_replay.map_snapshot_player_ids_to_engine` (sort by
    ``order`` field; first player → ``co_p0_id``, second → ``co_p1_id``).
    """
    players = first_snap.get("players") or {}
    rows: list[tuple[int, int]] = []
    for _k, p in players.items():
        if not isinstance(p, dict):
            continue
        try:
            o = int(p.get("order", 0))
            cid = int(p["co_id"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((o, cid))
    rows.sort(key=lambda t: t[0])
    if len(rows) < 2:
        return None
    c0, c1 = rows[0][1], rows[1][1]
    return {
        "games_id": int(games_id),
        "map_id": int(map_id),
        "co_p0_id": int(c0),
        "co_p1_id": int(c1),
        "tier": str(tier),
        "matchup": "",
    }
