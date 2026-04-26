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


def train_sampling_map_ids(map_pool_path: Path) -> set[int]:
    """
    Map ids that :class:`rl.env.AWBWEnv` can draw when sampling a new episode
    with the default pool file and **without** a per-run ``map_id`` filter.

    Mirrors ``AWBWEnv.__init__`` (``_sample_map_pool``): if the pool contains
    any ``type == \"std\"`` entries, sampling is uniform over those; otherwise
    it falls back to the full JSON list.

    Note: ``train.py`` / :class:`rl.self_play.SelfPlayTrainer` may pass
    ``map_pool`` restricted to a single ``map_id``; that runtime filter is not
    represented here.
    """
    with open(map_pool_path, encoding="utf-8") as f:
        pool: list[dict] = json.load(f)
    std = [m for m in pool if m.get("type") == "std"]
    use = std if std else pool
    return {int(m["map_id"]) for m in use}
