"""Regression tests for tools/replay_snapshot_compare.py bar conversion.

AWBW snapshot ``hit_points`` is the float internal_hp / 10. Both AWBW's UI and
``engine.unit.Unit.display_hp`` use **ceiling** to derive the displayed bar
(1..10). An earlier ``int(round(...))`` implementation produced spurious bar
mismatches against the engine for any non-integer ``hit_points`` whose
rounded value disagreed with its ceiling (e.g. 6.3 → round=6 vs ceil=7).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.diff_replay_zips import load_replay
from tools.replay_snapshot_compare import (
    _php_unit_bars,
    find_live_unit_overlaps,
    php_internal_from_snapshot_hit_points,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "hit_points, expected",
    [
        (None, 0),
        (0.0, 0),
        (0.1, 1),
        (1.0, 1),
        (4.4, 5),
        (4.9, 5),
        (6.0, 6),
        (6.3, 7),
        (6.5, 7),
        (7.1, 8),
        (8.2, 9),
        (10.0, 10),
        (10.5, 10),
        (-1.0, 0),
    ],
)
def test_php_unit_bars_uses_ceiling(hit_points, expected):
    assert _php_unit_bars({"hit_points": hit_points}) == expected


def test_php_unit_bars_matches_engine_display_hp_for_internal_hp_range():
    """For every internal HP 0..100, ceil(internal/10) must match both sides."""
    for internal_hp in range(0, 101):
        php_hp = round(internal_hp / 10.0, 1)
        engine_bars = (internal_hp + 9) // 10
        php_bars = _php_unit_bars({"hit_points": php_hp})
        assert php_bars == engine_bars == math.ceil(php_hp), (
            f"internal_hp={internal_hp} php={php_hp} engine={engine_bars} "
            f"php_bars={php_bars}"
        )


def test_php_internal_coerce_200scale_small_float():
    assert php_internal_from_snapshot_hit_points(0.1, 20) == 20
    assert php_internal_from_snapshot_hit_points(0.2, 40) == 40
    assert php_internal_from_snapshot_hit_points(6.3, 63) == 63


def test_php_internal_coerce_300scale_lossy_point_one():
    """GL tight zips sometimes store 0.1 for ~30 internal (0.15 true); ×300=30."""
    assert php_internal_from_snapshot_hit_points(0.1, 30) == 30
    assert php_internal_from_snapshot_hit_points(0.1, 27) == 30


def test_php_unit_bars_with_engine_coerces_01_to_two_bars():
    assert _php_unit_bars({"hit_points": 0.1}, engine_internal_hp=20) == 2


def test_find_live_unit_overlaps_ignores_carried_cargo():
    frames = [
        {
            "day": 1,
            "turn": 100001,
            "units": {
                0: {
                    "id": 1,
                    "players_id": 100001,
                    "name": "APC",
                    "x": 2,
                    "y": 3,
                    "hit_points": 10.0,
                    "carried": "N",
                },
                1: {
                    "id": 2,
                    "players_id": 100001,
                    "name": "Infantry",
                    "x": 2,
                    "y": 3,
                    "hit_points": 10.0,
                    "carried": "Y",
                },
            },
        }
    ]
    assert find_live_unit_overlaps(frames) == []


def test_find_live_unit_overlaps_reports_non_cargo_stack():
    frames = [
        {
            "day": 19,
            "turn": 100002,
            "units": {
                0: {
                    "id": 59,
                    "players_id": 100001,
                    "name": "Infantry",
                    "x": 13,
                    "y": 5,
                    "hit_points": 10.0,
                    "carried": "N",
                },
                1: {
                    "id": 36,
                    "players_id": 100002,
                    "name": "Recon",
                    "x": 13,
                    "y": 5,
                    "hit_points": 1.5,
                    "carried": "N",
                },
            },
        }
    ]
    overlaps = find_live_unit_overlaps(frames)
    assert len(overlaps) == 1
    assert overlaps[0].frame_index == 0
    assert overlaps[0].day == 19
    assert overlaps[0].turn == 100002
    assert overlaps[0].x == 13
    assert overlaps[0].y == 5
    assert overlaps[0].units == (
        (59, 100001, "Infantry", 10.0),
        (36, 100002, "Recon", 1.5),
    )


@pytest.mark.skipif(
    not (ROOT / "replays" / "524629.zip").is_file(),
    reason="524629 replay fixture not present",
)
def test_find_live_unit_overlaps_flags_524629_first_stack():
    overlaps = find_live_unit_overlaps(load_replay(ROOT / "replays" / "524629.zip"))

    assert overlaps
    first = overlaps[0]
    assert first.frame_index == 32
    assert first.day == 17
    assert first.turn == 100001
    assert (first.x, first.y) == (15, 10)
