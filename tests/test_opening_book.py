"""Opening book index + legal suggestion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl.opening_book import OpeningBookController, OpeningBookIndex, TwoSidedOpeningBookManager


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
    c.on_episode_start(episode_id=1, map_id=123858, co_id_for_seat=1, enabled=True)
    m = np.ones(35_000, dtype=bool)  # ACTION_SPACE_SIZE
    a = c.suggest_flat(calendar_turn=1, action_mask=m)
    assert a == 2
    a2 = c.suggest_flat(calendar_turn=1, action_mask=m)
    assert a2 == 3


def test_horizon_zero_no_day_cap_in_book(tmp_path: Path) -> None:
    p = tmp_path / "b.jsonl"
    book = {
        "book_id": "hz0",
        "map_id": 9,
        "seat": 0,
        "co_id": None,
        "horizon_days": 0,
        "action_indices": [1, 2],
    }
    p.write_text(json.dumps(book) + "\n", encoding="utf-8")
    idx = OpeningBookIndex.from_jsonl(p)
    c = OpeningBookController(
        idx,
        seat=0,
        strict_co=False,
        rng=__import__("random").Random(0),
        max_calendar_turn=None,
    )
    c.on_episode_start(episode_id=1, map_id=9, co_id_for_seat=None, enabled=True)
    m = np.ones(35_000, dtype=bool)
    assert c.suggest_flat(calendar_turn=50, action_mask=m) == 1
    assert c.suggest_flat(calendar_turn=50, action_mask=m) == 2
    assert c.suggest_flat(calendar_turn=50, action_mask=m) is None


def test_peek_does_not_advance_commit_does(tmp_path: Path) -> None:
    p = tmp_path / "b.jsonl"
    book = {
        "book_id": "peek",
        "map_id": 42,
        "seat": 0,
        "co_id": None,
        "horizon_days": 0,
        "action_indices": [10, 11],
    }
    p.write_text(json.dumps(book) + "\n", encoding="utf-8")
    idx = OpeningBookIndex.from_jsonl(p)
    c = OpeningBookController(
        idx, seat=0, strict_co=False, rng=__import__("random").Random(0), max_calendar_turn=None
    )
    c.on_episode_start(episode_id=1, map_id=42, co_id_for_seat=None, enabled=True)
    m = np.zeros(35_000, dtype=bool)
    m[10] = True
    m[11] = True
    assert c.peek_flat(calendar_turn=1, action_mask=m) == 10
    assert c.peek_flat(calendar_turn=1, action_mask=m) == 10
    c.commit_flat(10)
    assert c.peek_flat(calendar_turn=1, action_mask=m) == 11
    c.commit_flat(11)
    assert c.peek_flat(calendar_turn=1, action_mask=m) is None


def test_joint_schema_line(tmp_path: Path) -> None:
    p = tmp_path / "joint.jsonl"
    row = {
        "map_id": 99,
        "joint_book_id": "j1",
        "horizon_days": 0,
        "seats": {
            "0": {"action_indices": [1, 2]},
            "1": {"action_indices": [3, 4]},
        },
    }
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    idx = OpeningBookIndex.from_jsonl(p)
    assert len(idx.by_map_seat[(99, 0)]) == 1
    assert len(idx.by_map_seat[(99, 1)]) == 1
    assert idx.by_map_seat[(99, 0)][0].action_indices == [1, 2]
    assert idx.by_map_seat[(99, 1)][0].book_id.startswith("j1_s")


def test_ranked_human_openings_index_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "opening_books" / "ranked_std_human_openings.jsonl"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    idx = OpeningBookIndex.from_jsonl(path)
    map_seats = set(idx.by_map_seat.keys())
    assert map_seats
    for books in idx.by_map_seat.values():
        assert books
        for b in books:
            assert b.map_id > 0
            assert b.seat in (0, 1)
            assert b.action_indices


def test_two_sided_manager_independent_cursors(tmp_path: Path) -> None:
    p = tmp_path / "both.jsonl"
    lines = []
    for seat, actions in ((0, [100, 101]), (1, [200, 201])):
        lines.append(
            json.dumps(
                {
                    "book_id": f"m{seat}",
                    "map_id": 7,
                    "seat": seat,
                    "co_id": None,
                    "horizon_days": 0,
                    "action_indices": actions,
                }
            )
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mgr = TwoSidedOpeningBookManager(p, seats="both", prob=1.0, strict_co=False, seed=0)
    mgr.on_episode_start(episode_id=1, map_id=7, co_ids=[1, 2])
    m = np.zeros(35_000, dtype=bool)
    m[100] = m[101] = m[200] = m[201] = True
    assert mgr.peek_flat(seat=0, calendar_turn=1, action_mask=m) == 100
    assert mgr.peek_flat(seat=1, calendar_turn=1, action_mask=m) == 200
    mgr.suggest_flat(seat=0, calendar_turn=1, action_mask=m)
    assert mgr.peek_flat(seat=0, calendar_turn=1, action_mask=m) == 101
    assert mgr.peek_flat(seat=1, calendar_turn=1, action_mask=m) == 200
