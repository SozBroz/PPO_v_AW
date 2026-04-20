"""Drift-spawn recovery for ``oracle_move_no_unit`` (closes 1605367 / 1624281 / 1627324)."""

from __future__ import annotations

import unittest
from pathlib import Path

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.oracle_zip_replay import _oracle_drift_spawn_mover_from_global

ROOT = Path(__file__).resolve().parents[1]


def _make_state():
    md = load_map(
        159501,
        ROOT / "data" / "gl_map_pool.json",
        ROOT / "data" / "maps",
    )
    st = make_initial_state(
        md, 1, 2, starting_funds=9000, tier_name="T2", replay_first_mover=0
    )
    st.units[0] = []
    st.units[1] = []
    return st


class TestOracleMoveNoUnitDriftSpawn(unittest.TestCase):
    def test_spawn_on_empty_passable_path_start(self) -> None:
        """Engine lost the mover entirely; drift-spawn lands at path-start with AWBW id."""
        st = _make_state()
        gu = {
            "units_id": 192345678,
            "units_hit_points": 7,
            "units_players_id": 9,
            "units_name": "Infantry",
        }
        spawned = _oracle_drift_spawn_mover_from_global(
            st, eng=0, gu=gu,
            declared_mover_type=UnitType.INFANTRY,
            sr=4, sc=4, er=5, ec=4,
        )
        self.assertIsNotNone(spawned)
        assert spawned is not None  # narrow for type-checker
        self.assertEqual(spawned.unit_type, UnitType.INFANTRY)
        self.assertEqual(spawned.player, 0)
        self.assertEqual(spawned.pos, (4, 4))
        self.assertEqual(spawned.unit_id, 192345678)
        self.assertEqual(spawned.hp, 70)
        self.assertEqual(spawned.fuel, UNIT_STATS[UnitType.INFANTRY].max_fuel)

    def test_decline_on_friendly_non_carrier_at_path_start(self) -> None:
        """Friendly non-carrier at path-start: never overwrite — would mask a real bug."""
        st = _make_state()
        existing = Unit(
            unit_type=UnitType.TANK, player=0, hp=100, ammo=9, fuel=70,
            pos=(4, 4), moved=False, loaded_units=[], is_submerged=False,
            capture_progress=20, unit_id=11,
        )
        st.units[0].append(existing)
        gu = {
            "units_id": 192345678,
            "units_players_id": 9,
            "units_name": "Mech",
        }
        spawned = _oracle_drift_spawn_mover_from_global(
            st, eng=0, gu=gu,
            declared_mover_type=UnitType.MECH,
            sr=4, sc=4, er=5, ec=4,
        )
        self.assertIsNone(spawned)
        self.assertIs(st.get_unit_at(4, 4), existing)

    def test_teleport_friendly_carrier_to_empty_path_end_then_spawn_cargo(self) -> None:
        """``Load`` envelope drift: friendly carrier on path-start, empty path-end.

        Ground truth (game 1632702): T-Copter sits at path-start in the engine,
        AWBW says T-Copter is at path-end and Mech (cargo) at path-start moving
        into it. Recovery teleports the T-Copter to path-end and spawns the
        Mech at path-start.
        """
        st = _make_state()
        carrier = Unit(
            unit_type=UnitType.T_COPTER, player=0, hp=100,
            ammo=UNIT_STATS[UnitType.T_COPTER].max_ammo,
            fuel=UNIT_STATS[UnitType.T_COPTER].max_fuel,
            pos=(4, 4), moved=False, loaded_units=[], is_submerged=False,
            capture_progress=20, unit_id=42,
        )
        st.units[0].append(carrier)
        gu = {
            "units_id": 192700001,
            "units_players_id": 9,
            "units_name": "Mech",
        }
        spawned = _oracle_drift_spawn_mover_from_global(
            st, eng=0, gu=gu,
            declared_mover_type=UnitType.MECH,
            sr=4, sc=4, er=5, ec=4,
        )
        self.assertIsNotNone(spawned)
        assert spawned is not None
        self.assertEqual(spawned.unit_type, UnitType.MECH)
        self.assertEqual(spawned.pos, (4, 4))
        self.assertEqual(carrier.pos, (5, 4))
        self.assertTrue(carrier.moved)

    def test_clear_enemy_ghost_at_path_end(self) -> None:
        """Enemy ghost at path-end (combat drift): zero HP so engine path-walker accepts."""
        st = _make_state()
        ghost = Unit(
            unit_type=UnitType.INFANTRY, player=1, hp=10, ammo=0, fuel=99,
            pos=(5, 4), moved=False, loaded_units=[], is_submerged=False,
            capture_progress=20, unit_id=77,
        )
        st.units[1].append(ghost)
        gu = {
            "units_id": 192345679,
            "units_players_id": 9,
            "units_name": "Infantry",
        }
        spawned = _oracle_drift_spawn_mover_from_global(
            st, eng=0, gu=gu,
            declared_mover_type=UnitType.INFANTRY,
            sr=4, sc=4, er=5, ec=4,
        )
        self.assertIsNotNone(spawned)
        self.assertEqual(ghost.hp, 0)


if __name__ == "__main__":
    unittest.main()
