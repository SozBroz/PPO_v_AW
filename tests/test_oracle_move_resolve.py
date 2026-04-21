"""Regression for plain ``Move`` path-end reconciliation (oracle_zip_replay)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.action import ActionStage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import UnsupportedOracleAction, apply_oracle_action_json


class TestOracleMoveResolve(unittest.TestCase):
    def test_plain_move_forces_zip_path_end_when_reachability_omits_tail_gl_1607045_shape(
        self,
    ) -> None:
        """GL 1607045 / Phase 9 Lane L: ZIP ``paths.global`` tail can be missing from
        ``compute_reachable_costs`` in the re-simulated state while AWBW still records
        the walk onto that tile.

        Without reconciliation the engine stops at ``_nearest_reachable_along_path`` and
        the post-terminator invariant raises ``Move: engine truncated path…``.
        """
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        ist = UNIT_STATS[UnitType.INFANTRY]
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                100,
                ist.max_ammo,
                ist.max_fuel,
                (5, 5),
                False,
                [],
                False,
                20,
                190854762,
            )
        )
        s.active_player = 0
        s.action_stage = ActionStage.SELECT

        awbw_pid = 3716287
        move = {
            "action": "Move",
            "unit": {
                "global": {
                    "units_id": 190854762,
                    "units_players_id": awbw_pid,
                    "units_name": "Infantry",
                    "units_y": 5,
                    "units_x": 5,
                    "units_movement_points": 3,
                    "units_vision": 1,
                    "units_fuel": 99,
                    "units_fuel_per_turn": 0,
                    "units_sub_dive": "N",
                    "units_ammo": 0,
                    "units_short_range": 0,
                    "units_long_range": 0,
                    "units_second_weapon": "N",
                    "units_symbol": "G",
                    "units_cost": 1000,
                    "units_movement_type": "F",
                    "units_moved": 0,
                    "units_capture": 0,
                    "units_fired": 0,
                    "units_hit_points": 10,
                    "units_cargo1_units_id": 0,
                    "units_cargo2_units_id": 0,
                    "units_carried": "N",
                    "countries_code": "os",
                }
            },
            "paths": {
                "global": [
                    {"y": 5, "x": 5},
                    {"y": 5, "x": 6},
                    {"y": 5, "x": 7},
                ]
            },
        }

        def fake_costs(_st: object, _u: object) -> dict[tuple[int, int], int]:
            return {(5, 5): 0, (5, 6): 1}

        with patch("tools.oracle_zip_replay.compute_reachable_costs", fake_costs):
            apply_oracle_action_json(
                s,
                move,
                {awbw_pid: 0},
                envelope_awbw_player_id=awbw_pid,
            )

        u = s.units[0][0]
        self.assertEqual((int(u.pos[0]), int(u.pos[1])), (5, 7))

    def test_plain_move_truncation_still_raises_when_tail_occupied_by_other_unit(
        self,
    ) -> None:
        """Do not ``_move_unit_forced`` onto a tile held by another live unit."""
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        ist = UNIT_STATS[UnitType.INFANTRY]
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                100,
                ist.max_ammo,
                ist.max_fuel,
                (5, 5),
                False,
                [],
                False,
                20,
                1,
            )
        )
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                100,
                ist.max_ammo,
                ist.max_fuel,
                (5, 7),
                False,
                [],
                False,
                20,
                2,
            )
        )
        s.active_player = 0
        s.action_stage = ActionStage.SELECT

        awbw_pid = 99999
        move = {
            "action": "Move",
            "unit": {
                "global": {
                    "units_id": 1,
                    "units_players_id": awbw_pid,
                    "units_name": "Infantry",
                    "units_y": 5,
                    "units_x": 5,
                    "units_movement_points": 3,
                    "units_vision": 1,
                    "units_fuel": 99,
                    "units_fuel_per_turn": 0,
                    "units_sub_dive": "N",
                    "units_ammo": 0,
                    "units_short_range": 0,
                    "units_long_range": 0,
                    "units_second_weapon": "N",
                    "units_symbol": "G",
                    "units_cost": 1000,
                    "units_movement_type": "F",
                    "units_moved": 0,
                    "units_capture": 0,
                    "units_fired": 0,
                    "units_hit_points": 10,
                    "units_cargo1_units_id": 0,
                    "units_cargo2_units_id": 0,
                    "units_carried": "N",
                    "countries_code": "os",
                }
            },
            "paths": {
                "global": [
                    {"y": 5, "x": 5},
                    {"y": 5, "x": 6},
                    {"y": 5, "x": 7},
                ]
            },
        }

        def fake_costs(_st: object, _u: object) -> dict[tuple[int, int], int]:
            return {(5, 5): 0, (5, 6): 1}

        with (
            patch("tools.oracle_zip_replay.compute_reachable_costs", fake_costs),
            self.assertRaises(UnsupportedOracleAction) as ctx,
        ):
            apply_oracle_action_json(
                s,
                move,
                {awbw_pid: 0},
                envelope_awbw_player_id=awbw_pid,
            )
        self.assertIn("engine truncated path", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
