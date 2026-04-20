"""Black Boat repair oracle uses ``selected_move_pos`` as the adjacency anchor (ACTION stage)."""

from __future__ import annotations

import copy
import unittest

from engine.action import ActionStage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import (
    _black_boat_oracle_action_tile,
    _oracle_attack_eval_pos,
    _oracle_snap_active_player_to_engine,
    _repair_repaired_global_dict,
)


class TestOracleRepairTile(unittest.TestCase):
    def test_black_boat_action_tile_prefers_selected_move_pos(self) -> None:
        m = load_map(126428, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 14, 21, tier_name="T4", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        st = UNIT_STATS[UnitType.BLACK_BOAT]
        boat = Unit(
            UnitType.BLACK_BOAT,
            1,
            100,
            st.max_ammo,
            st.max_fuel,
            (0, 12),
            False,
            [],
            False,
            20,
            1,
        )
        s.units[1].append(boat)
        s.active_player = 1
        s.action_stage = ActionStage.ACTION
        s.selected_unit = boat
        s.selected_move_pos = (0, 13)
        self.assertEqual(_black_boat_oracle_action_tile(s, boat), (0, 13))
        s.selected_move_pos = None
        self.assertEqual(_black_boat_oracle_action_tile(s, boat), (0, 12))

    def test_black_boat_action_tile_matches_attack_eval_not_selection_identity(self) -> None:
        """``selected_unit`` may not be the same object as the boat in ``units[]`` (AWBW paths)."""
        m = load_map(126428, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 14, 21, tier_name="T4", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        st = UNIT_STATS[UnitType.BLACK_BOAT]
        boat = Unit(
            UnitType.BLACK_BOAT,
            1,
            100,
            st.max_ammo,
            st.max_fuel,
            (0, 12),
            False,
            [],
            False,
            20,
            1,
        )
        s.units[1].append(boat)
        s.active_player = 1
        s.action_stage = ActionStage.ACTION
        s.selected_unit = copy.copy(boat)
        s.selected_move_pos = (0, 13)
        self.assertIsNot(s.selected_unit, boat)
        self.assertEqual(_black_boat_oracle_action_tile(s, boat), (0, 13))
        self.assertEqual(_oracle_attack_eval_pos(s, boat), (0, 13))

    def test_repaired_global_int_id(self) -> None:
        d = _repair_repaired_global_dict({"repaired": {"global": 191234567}})
        self.assertEqual(d, {"units_id": 191234567})

    def test_repaired_global_flat_units_id(self) -> None:
        d = _repair_repaired_global_dict({"repaired": {"units_id": 42, "units_hit_points": 5}})
        self.assertEqual(d.get("units_id"), 42)
        self.assertEqual(d.get("units_hit_points"), 5)

    def test_repaired_global_seat_bucket_gl(
        self,
    ) -> None:
        """games_id cluster: repaired snapshot only under envelope seat key (oracle_repair)."""
        d = _repair_repaired_global_dict(
            {
                "repaired": {
                    "987001": {"units_id": 1623866, "units_hit_points": 8},
                },
            },
            envelope_awbw_player_id=987001,
        )
        self.assertEqual(d.get("units_id"), 1623866)

    def test_oracle_snap_active_player_to_engine_keeps_boat_owner(
        self,
    ) -> None:
        """Repair no-path selected Black Boat vs wrong active_player counter (oracle_repair)."""
        m = load_map(126428, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 14, 21, tier_name="T4", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        st = UNIT_STATS[UnitType.BLACK_BOAT]
        boat = Unit(
            UnitType.BLACK_BOAT,
            1,
            100,
            st.max_ammo,
            st.max_fuel,
            (0, 12),
            False,
            [],
            False,
            20,
            1,
        )
        s.units[1].append(boat)
        s.active_player = 0
        s.action_stage = ActionStage.ACTION
        s.selected_unit = boat
        s.selected_move_pos = (0, 12)
        awbw_map = {101: 0, 102: 1}
        _oracle_snap_active_player_to_engine(s, 1, awbw_map, None)
        self.assertEqual(int(s.active_player), 1)


if __name__ == "__main__":
    unittest.main()
