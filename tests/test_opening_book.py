"""Opening book index + legal suggestion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl.opening_book import OpeningBookController, OpeningBookIndex


def test_index_loads_and_samples(tmp_path: Path) -> None:
    p = tmp_path / "b.jsonl"
    book = {
        "book_id": "test1",
        "map_id": 123858,
        "seat": 1,
        "co_id": 1,
        "horizon_days": 3,
        "action_indices": [2, 3, 4],
    }
    p.write_text(json.dumps(book) + "\n", encoding="utf-8")
    idx = OpeningBookIndex.from_jsonl(p)
    assert (123858, 1) in idx.by_map_seat
    c = OpeningBookController(
        idx, seat=1, strict_co=False, rng=__import__("random").Random(0), max_calendar_turn=3
    )
    c.on_episode_start(episode_id=1, map_id=123858, co_id_for_seat=1)
    m = np.ones(35_000, dtype=bool)  # ACTION_SPACE_SIZE
    a = c.suggest_flat(calendar_turn=1, action_mask=m)
    assert a == 2
    a2 = c.suggest_flat(calendar_turn=1, action_mask=m)
    assert a2 == 3
