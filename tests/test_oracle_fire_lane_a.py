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
    _oracle_move_med_tank_label_engine_tank_drift,
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

    def test_move_med_tank_label_engine_tank_drift_gl1607045(self) -> None:
        """1607045: zip ``Md.Tank`` + PHP id while engine holds ``TANK`` (no ``MED_TANK``)."""
        md = load_map(77060, POOL, MAPS)
        s = make_initial_state(
            md, 5, 28, tier_name="T3", starting_funds=0, replay_first_mover=0
        )
        s.units[0] = []
        s.units[1] = []
        st = UNIT_STATS[UnitType.TANK]
        mover = Unit(
            UnitType.TANK,
            1,
            30,
            st.max_ammo,
            st.max_fuel,
            (18, 13),
            False,
            [],
            False,
            30,
            999001,
        )
        other = Unit(
            UnitType.TANK,
            1,
            80,
            st.max_ammo,
            st.max_fuel,
            (16, 11),
            False,
            [],
            False,
            80,
            999002,
        )
        s.units[1].extend((mover, other))
        paths = [{"y": 18, "x": 13}, {"y": 18, "x": 12}, {"y": 18, "x": 11}]
        gu = {
            "units_id": 191018230,
            "units_players_id": 3716287,
            "units_name": "Md.Tank",
            "units_y": 18,
            "units_x": 11,
            "units_hit_points": 3,
        }
        u = _oracle_move_med_tank_label_engine_tank_drift(
            s,
            1,
            UnitType.MED_TANK,
            paths,
            gu,
            (18, 13),
            (18, 11),
            (18, 11),
        )
        self.assertIsNotNone(u)
        self.assertEqual(u.unit_id, 999001)
        # Two tanks on a bent path: only path-start cell disambiguates.
        mover.pos = (18, 11)
        mover.hp = 70
        other.pos = (16, 11)
        other.hp = 80
        paths2 = [
            {"y": 18, "x": 11},
            {"y": 17, "x": 11},
            {"y": 16, "x": 11},
            {"y": 15, "x": 11},
            {"y": 14, "x": 11},
            {"y": 14, "x": 12},
        ]
        gu2 = dict(gu)
        gu2["units_hit_points"] = 10
        gu2["units_y"] = 14
        gu2["units_x"] = 12
        u2 = _oracle_move_med_tank_label_engine_tank_drift(
            s,
            1,
            UnitType.MED_TANK,
            paths2,
            gu2,
            (18, 11),
            (14, 12),
            (14, 12),
        )
        self.assertIsNotNone(u2)
        self.assertEqual(u2.unit_id, 999001)

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
