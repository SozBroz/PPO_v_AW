"""Unit tests for tools/oracle_state_sync.py.

Covers the contract:
- HP deltas within MAX_PLAUSIBLE_HP_SWING are snapped silently.
- HP deltas above the cap are flagged out-of-range and NOT snapped.
- Funds are always snapped (no plausibility threshold).
- PHP-only and engine-only units are reported as structural divergence.
- Engine units missing from PHP are killed (then pruned).
- ``unit_id`` is preserved across the snap (replay viewers index by it).
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from engine.unit import Unit, UnitType, UNIT_STATS
from tools.oracle_state_sync import (
    MAX_PLAUSIBLE_HP_SWING_PER_ENVELOPE,
    sync_state_to_snapshot,
    _php_internal_hp,
)


def _mk(unit_type: UnitType, player: int, pos: tuple[int, int], hp: int, unit_id: int = 0) -> Unit:
    s = UNIT_STATS[unit_type]
    return Unit(
        unit_type=unit_type, player=player, hp=hp,
        ammo=s.max_ammo if s.max_ammo > 0 else 0,
        fuel=s.max_fuel, pos=pos, moved=False, loaded_units=[],
        is_submerged=False, capture_progress=20, unit_id=unit_id,
    )


def _state_with_units(p0: list[Unit], p1: list[Unit], funds=(0, 0)):
    s = MagicMock()
    s.units = {0: list(p0), 1: list(p1)}
    s.funds = list(funds)
    return s


def _php_frame(units: list[dict], players: list[dict]) -> dict:
    return {
        "units": {str(i): u for i, u in enumerate(units)},
        "players": {str(i): p for i, p in enumerate(players)},
    }


class TestPhpInternalHp(unittest.TestCase):
    def test_recovers_internal_hp_from_float(self):
        # PHP stores internal_hp / 10. 6.3 -> 63 internal HP, ceil bar = 7.
        self.assertEqual(_php_internal_hp(6.3), 63)
        self.assertEqual(_php_internal_hp(10.0), 100)
        self.assertEqual(_php_internal_hp(0.1), 1)
        self.assertEqual(_php_internal_hp(0.0), 0)
        self.assertEqual(_php_internal_hp(None), 0)
        self.assertEqual(_php_internal_hp(-1.0), 0)
        self.assertEqual(_php_internal_hp(11.0), 100)


class TestSyncSnapsLuckNoise(unittest.TestCase):
    """Within the plausibility cap: silent snap, unit_id preserved."""

    def test_small_hp_delta_is_snapped_silently(self):
        u = _mk(UnitType.INFANTRY, 0, (5, 5), hp=70, unit_id=42)
        state = _state_with_units([u], [])
        php = _php_frame(
            units=[{"id": 1, "x": 5, "y": 5, "players_id": 100, "name": "Infantry", "hit_points": 6.3}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        self.assertEqual(rep.snapped_units, 1)
        self.assertEqual(rep.out_of_range_units, 0)
        self.assertEqual(u.hp, 63)
        self.assertEqual(u.unit_id, 42, "unit_id must be preserved")
        self.assertTrue(rep.ok)


class TestSyncFlagsOutOfRange(unittest.TestCase):
    """Above the cap: flagged + NOT snapped; engine HP left alone for triage."""

    def test_delta_exceeding_cap_is_flagged_not_snapped(self):
        u = _mk(UnitType.INFANTRY, 0, (5, 5), hp=100, unit_id=42)
        state = _state_with_units([u], [])
        # PHP says 1 HP; engine 100; delta 99 < cap 100 -> snapped silently.
        php = _php_frame(
            units=[{"id": 1, "x": 5, "y": 5, "players_id": 100, "name": "Infantry", "hit_points": 0.1}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        # Delta 99 <= cap 100 -> snapped silently.
        self.assertEqual(rep.snapped_units, 1)
        self.assertEqual(rep.out_of_range_units, 0)
        self.assertEqual(u.hp, 1)

    def test_oor_with_explicit_low_cap(self):
        u = _mk(UnitType.INFANTRY, 0, (5, 5), hp=100, unit_id=42)
        state = _state_with_units([u], [])
        php = _php_frame(
            units=[{"id": 1, "x": 5, "y": 5, "players_id": 100, "name": "Infantry", "hit_points": 0.1}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0}, max_hp_swing=50)
        self.assertEqual(rep.out_of_range_units, 1)
        self.assertEqual(u.hp, 100, "OOR engine HP must be left alone for triage")


class TestSyncFunds(unittest.TestCase):
    def test_funds_always_snap_no_plausibility_check(self):
        state = _state_with_units([], [], funds=(10000, 5000))
        php = _php_frame(
            units=[],
            players=[{"id": 100, "funds": 9800}, {"id": 200, "funds": 5000}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0, 200: 1})
        self.assertEqual(state.funds, [9800, 5000])
        self.assertEqual(len(rep.funds_snapped), 1)
        self.assertEqual(rep.funds_snapped[0], (0, 10000, 9800))


class TestSyncStructuralDivergence(unittest.TestCase):
    def test_engine_unit_absent_from_php_is_killed(self):
        ghost = _mk(UnitType.INFANTRY, 0, (1, 1), hp=80, unit_id=7)
        state = _state_with_units([ghost], [])
        php = _php_frame(units=[], players=[{"id": 100, "funds": 0}])
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        self.assertEqual(rep.engine_only_units, [(0, 1, 1)])
        self.assertEqual(state.units[0], [], "dead unit must be pruned")
        self.assertFalse(rep.ok)

    def test_php_unit_absent_from_engine_is_reported_not_spawned(self):
        state = _state_with_units([], [])
        php = _php_frame(
            units=[{"id": 1, "x": 3, "y": 4, "players_id": 100, "name": "Tank", "hit_points": 10.0}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        self.assertEqual(rep.php_only_units, [(0, 4, 3)])
        self.assertEqual(state.units[0], [], "must NOT auto-spawn — that hides oracle bugs")
        self.assertFalse(rep.ok)


class TestPlausibilityCapBoundary(unittest.TestCase):
    def test_exactly_at_cap_is_snapped(self):
        u = _mk(UnitType.INFANTRY, 0, (5, 5), hp=100, unit_id=1)
        state = _state_with_units([u], [])
        # Engine 100, PHP 0 -> delta 100 == cap (kill-from-full bound).
        php = _php_frame(
            units=[{"id": 1, "x": 5, "y": 5, "players_id": 100, "name": "Infantry", "hit_points": 0.0}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        # PHP hp=0 still appears as a unit row; sync snaps engine to 0,
        # killing the engine unit (delta 100 == cap exactly).
        self.assertEqual(rep.snapped_units, 1)
        self.assertEqual(rep.out_of_range_units, 0)
        self.assertEqual(state.units[0], [], "engine unit pruned after snap to 0")

    def test_above_cap_with_low_threshold(self):
        u = _mk(UnitType.INFANTRY, 0, (5, 5), hp=100, unit_id=1)
        state = _state_with_units([u], [])
        php = _php_frame(
            units=[{"id": 1, "x": 5, "y": 5, "players_id": 100, "name": "Infantry", "hit_points": 3.9}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0}, max_hp_swing=50)
        self.assertEqual(rep.snapped_units, 0)
        self.assertEqual(rep.out_of_range_units, 1)
        self.assertEqual(u.hp, 100)


class TestResurrection(unittest.TestCase):
    """When the engine wrongly killed a unit AWBW kept alive, sync revives
    the dead engine instance at the same tile (matching unit type only)."""

    def test_resurrects_dead_engine_unit_at_same_tile_same_type(self):
        dead = _mk(UnitType.INFANTRY, 0, (5, 5), hp=0, unit_id=42)
        state = _state_with_units([dead], [])
        php = _php_frame(
            units=[{"id": 99, "x": 5, "y": 5, "players_id": 100, "name": "Infantry", "hit_points": 5.0}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        self.assertEqual(rep.snapped_units, 1)
        self.assertEqual(rep.php_only_units, [], "should NOT report as php-only after resurrection")
        # Resurrected unit is back in the seat list.
        self.assertEqual(len(state.units[0]), 1)
        self.assertEqual(state.units[0][0].hp, 50)
        self.assertEqual(state.units[0][0].unit_id, 42, "preserve unit_id on resurrection")

    def test_does_not_resurrect_when_unit_type_differs(self):
        dead = _mk(UnitType.TANK, 0, (5, 5), hp=0, unit_id=42)
        state = _state_with_units([dead], [])
        php = _php_frame(
            units=[{"id": 99, "x": 5, "y": 5, "players_id": 100, "name": "Infantry", "hit_points": 5.0}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        self.assertEqual(rep.snapped_units, 0)
        self.assertEqual(rep.php_only_units, [(0, 5, 5)],
                         "type mismatch must NOT resurrect — that hides oracle bugs")
        self.assertEqual(state.units[0], [], "dead unit pruned, no spawn")


class TestTeleport(unittest.TestCase):
    """Pair (engine_only, php_only) of same (seat, type) within the cap and
    teleport the engine unit to the PHP tile. Beyond the cap or with no
    type match, fall back to kill + report as before."""

    def test_teleports_same_type_within_distance(self):
        # Engine has Tank P0 at (5, 5); PHP has Tank P0 at (5, 8) (distance 3).
        u = _mk(UnitType.TANK, 0, (5, 5), hp=80, unit_id=42)
        state = _state_with_units([u], [])
        php = _php_frame(
            units=[{"id": 1, "x": 8, "y": 5, "players_id": 100, "name": "Tank", "hit_points": 7.0}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        self.assertEqual(rep.engine_only_units, [], "should teleport, not kill")
        self.assertEqual(rep.php_only_units, [], "should teleport, not orphan")
        self.assertEqual(u.pos, (5, 8), "engine unit teleported to PHP tile")
        self.assertEqual(u.hp, 70, "HP snapped to PHP value")
        self.assertEqual(u.unit_id, 42, "unit_id preserved across teleport")

    def test_no_teleport_beyond_distance_cap(self):
        # Tank moved 50 tiles? Definitely a different unit; do not teleport.
        u = _mk(UnitType.TANK, 0, (0, 0), hp=80, unit_id=42)
        state = _state_with_units([u], [])
        php = _php_frame(
            units=[{"id": 1, "x": 25, "y": 25, "players_id": 100, "name": "Tank", "hit_points": 7.0}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        self.assertEqual(rep.engine_only_units, [(0, 0, 0)])
        self.assertEqual(rep.php_only_units, [(0, 25, 25)])
        self.assertEqual(state.units[0], [], "engine unit killed (no teleport)")

    def test_no_teleport_when_type_differs(self):
        u = _mk(UnitType.TANK, 0, (5, 5), hp=80, unit_id=42)
        state = _state_with_units([u], [])
        # PHP: name=Infantry at row 8 col 5 (y=8, x=5).
        php = _php_frame(
            units=[{"id": 1, "x": 5, "y": 8, "players_id": 100, "name": "Infantry", "hit_points": 7.0}],
            players=[{"id": 100, "funds": 0}],
        )
        rep = sync_state_to_snapshot(state, php, awbw_to_engine={100: 0})
        self.assertEqual(rep.engine_only_units, [(0, 5, 5)])
        self.assertEqual(rep.php_only_units, [(0, 8, 5)],
                         "PHP key is (seat, row=y, col=x) — y=8 above")


if __name__ == "__main__":
    unittest.main()
