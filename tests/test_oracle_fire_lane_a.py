"""Lane A (oldest 15 GL ``oracle_fire`` targets) — regression guards.

Canonical GL IDs in docstrings:

* **1621898**, **1629921** — NeoTank vs B-Copter requires non-null ``get_base_damage``
  (``data/damage_table.json`` MG cells).
* **1625784** — :func:`tools.oracle_zip_replay._oracle_fire_no_path_snap_foot_unit_neighbor_to_empty_awbw_anchor`.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from engine.action import ActionStage
from engine.combat import get_base_damage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.oracle_zip_replay import (
    _oracle_fire_no_path_snap_foot_unit_neighbor_to_empty_awbw_anchor,
)

POOL = Path(__file__).resolve().parents[1] / "data" / "gl_map_pool.json"
MAPS = Path(__file__).resolve().parents[1] / "data" / "maps"


class TestOracleFireLaneA(unittest.TestCase):
    def test_neotank_bcopter_damage_table_gl1621898_gl1629921(self) -> None:
        self.assertEqual(get_base_damage(UnitType.NEO_TANK, UnitType.B_COPTER), 35)
        self.assertEqual(get_base_damage(UnitType.NEO_TANK, UnitType.T_COPTER), 35)
        self.assertEqual(get_base_damage(UnitType.MED_TANK, UnitType.B_COPTER), 35)
        self.assertIsNone(get_base_damage(UnitType.INFANTRY, UnitType.B_COPTER))

    def test_foot_snap_single_neighbor_gl1625784(self) -> None:
        md = load_map(69201, POOL, MAPS)
        s = make_initial_state(
            md, 1, 10, tier_name="T2", starting_funds=0, replay_first_mover=0
        )
        s.units[0] = []
        s.units[1] = []
        ist = UNIT_STATS[UnitType.INFANTRY]
        inf = Unit(
            UnitType.INFANTRY,
            0,
            60,
            ist.max_ammo,
            ist.max_fuel,
            (5, 9),
            False,
            [],
            False,
            60,
            52001,
        )
        foe = Unit(
            UnitType.INFANTRY,
            1,
            40,
            ist.max_ammo,
            ist.max_fuel,
            (5, 11),
            False,
            [],
            False,
            40,
            52002,
        )
        s.units[0].append(inf)
        s.units[1].append(foe)
        s.active_player = 0
        s.action_stage = ActionStage.SELECT
        self.assertTrue(
            _oracle_fire_no_path_snap_foot_unit_neighbor_to_empty_awbw_anchor(
                s,
                eng=0,
                awbw_units_id=999999,
                anchor_r=5,
                anchor_c=10,
                target_r=5,
                target_c=11,
                hp_hint=None,
            )
        )
        self.assertEqual(s.get_unit_at(5, 10), inf)

    def test_lane_a_gid_list_frozen(self) -> None:
        """Frozen roster for merge coordination (lane B/C owned elsewhere)."""
        lane_a = frozenset(
            {
                1607045,
                1615143,
                1615231,
                1617442,
                1621898,
                1624421,
                1625784,
                1626655,
                1627054,
                1627245,
                1627523,
                1628190,
                1629092,
                1629512,
                1629921,
            }
        )
        self.assertEqual(len(lane_a), 15)


if __name__ == "__main__":
    unittest.main()
