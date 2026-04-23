"""Phase 2a: precomputed terrain_id -> category table matches uncached ground truth."""
from __future__ import annotations

from engine.terrain import TERRAIN_TABLE
from rl.encoder import (
    TERRAIN_CATEGORIES,
    _TERRAIN_CATEGORY_TABLE,
    _compute_terrain_category_uncached,
    _get_terrain_category,
)

_VALID_CATEGORY_INDICES = set(TERRAIN_CATEGORIES.values())


def test_table_matches_function_for_all_known_ids() -> None:
    for tid in TERRAIN_TABLE.keys():
        assert _get_terrain_category(tid) == _compute_terrain_category_uncached(
            tid
        ), f"mismatch for terrain_id={tid}"


def test_unknown_terrain_id_falls_through() -> None:
    unknown = 99999
    assert unknown not in TERRAIN_TABLE
    assert _get_terrain_category(unknown) == _compute_terrain_category_uncached(unknown)


def test_table_is_built_at_import() -> None:
    assert len(_TERRAIN_CATEGORY_TABLE) > 0
    for v in _TERRAIN_CATEGORY_TABLE.values():
        assert v in _VALID_CATEGORY_INDICES
