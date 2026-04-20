"""
Global League **standard** map ids from ``data/gl_map_pool.json``.

Only entries with ``\"type\": \"std\"`` are included — the current GL Std rotation
from ``listMaps`` (see ``data/fetch_awbw.py``). Maps used in completed-game
scrapes but not in that rotation must not be replayed or validated as GL-Std.
"""
from __future__ import annotations

import json
from pathlib import Path


def gl_std_map_ids(map_pool_path: Path) -> set[int]:
    with open(map_pool_path, encoding="utf-8") as f:
        pool: list[dict] = json.load(f)
    return {int(m["map_id"]) for m in pool if m.get("type") == "std"}
