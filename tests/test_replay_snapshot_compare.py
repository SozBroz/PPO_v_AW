"""Regression tests for tools/replay_snapshot_compare.py bar conversion.

AWBW snapshot ``hit_points`` is the float internal_hp / 10. Both AWBW's UI and
``engine.unit.Unit.display_hp`` use **ceiling** to derive the displayed bar
(1..10). An earlier ``int(round(...))`` implementation produced spurious bar
mismatches against the engine for any non-integer ``hit_points`` whose
rounded value disagreed with its ceiling (e.g. 6.3 → round=6 vs ceil=7).
"""
from __future__ import annotations

import math

import pytest

from tools.replay_snapshot_compare import _php_unit_bars


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
