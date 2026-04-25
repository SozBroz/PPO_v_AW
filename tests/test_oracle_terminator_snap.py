"""Phase 10B — path-end snap for JOIN / LOAD / nested-Fire tails (oracle_zip_replay)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.action import ActionStage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import (
    apply_oracle_action_json,
    _oracle_path_tail_is_friendly_load_boarding,
    _oracle_path_tail_occupant_allows_forced_snap,
)


def _base_global(uid: int, awbw_pid: int, y: int, x: int, name: str = "Infantry") -> dict:
    return {
        "units_id": uid,
        "units_players_id": awbw_pid,
        "units_name": name,
        "units_y": y,
        "units_x": x,
        "units_movement_points": 3,
        "units_vision": 2,
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


class TestOraclePathTailSnapHelper(unittest.TestCase):
    def test_helper_accepts_load_onto_friendly_transport(self) -> None:
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        ist = UNIT_STATS[UnitType.INFANTRY]
        ast = UNIT_STATS[UnitType.APC]
        mover = Unit(
            UnitType.INFANTRY, 0, 100, ist.max_ammo, ist.max_fuel, (6, 8), False, [], False, 20, 1
        )
        apc = Unit(
            UnitType.APC, 0, 100, ast.max_ammo, ast.max_fuel, (6, 10), False, [], False, 20, 2
        )
        s.units[0].extend([mover, apc])
        self.assertTrue(
            _oracle_path_tail_is_friendly_load_boarding(s, mover, (6, 10), engine_player=0)
        )


class TestOracleTerminatorEnvelopeSnap(unittest.TestCase):
    """ZIP tail missing from reach: reconcile before JOIN/LOAD/CAPTURE terminators."""

    def test_join_nested_move_snaps_when_reach_omits_partner_tile(self) -> None:
        """AWBW path ends on join partner; engine reach can omit occupied tail."""
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
                (6, 8),
                False,
                [],
                False,
                20,
                201,
            )
        )
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                50,
                ist.max_ammo,
                ist.max_fuel,
                (6, 10),
                False,
                [],
                False,
                20,
                202,
            )
        )
        s.active_player = 0
        s.action_stage = ActionStage.SELECT

        awbw_pid = 3716287
        join_env = {
            "action": "Join",
            "Move": {
                "unit": {"global": _base_global(201, awbw_pid, 6, 8)},
                "paths": {"global": [{"y": 6, "x": 8}, {"y": 6, "x": 9}, {"y": 6, "x": 10}]},
            },
        }

        def fake_costs(_st: object, _u: object) -> dict[tuple[int, int], int]:
            return {(6, 8): 0, (6, 9): 1}

        with patch("tools.oracle_zip_replay.compute_reachable_costs", fake_costs):
            apply_oracle_action_json(
                s,
                join_env,
                {awbw_pid: 0},
                envelope_awbw_player_id=awbw_pid,
            )

        positions = {(int(u.pos[0]), int(u.pos[1])) for u in s.units[0] if u.is_alive}
        self.assertNotIn((6, 8), positions)
        self.assertIn((6, 10), positions)

    def test_load_nested_move_snaps_when_reach_omits_transport_tile(self) -> None:
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        ist = UNIT_STATS[UnitType.INFANTRY]
        ast = UNIT_STATS[UnitType.APC]
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                100,
                ist.max_ammo,
                ist.max_fuel,
                (6, 8),
                False,
                [],
                False,
                20,
                301,
            )
        )
        s.units[0].append(
            Unit(
                UnitType.APC,
                0,
                100,
                ast.max_ammo,
                ast.max_fuel,
                (6, 10),
                False,
                [],
                False,
                20,
                302,
            )
        )
        s.active_player = 0
        s.action_stage = ActionStage.SELECT

        awbw_pid = 3716288
        load_env = {
            "action": "Load",
            "Move": {
                "unit": {"global": _base_global(301, awbw_pid, 6, 8)},
                "paths": {"global": [{"y": 6, "x": 8}, {"y": 6, "x": 9}, {"y": 6, "x": 10}]},
            },
        }

        def fake_costs(_st: object, _u: object) -> dict[tuple[int, int], int]:
            return {(6, 8): 0, (6, 9): 1}

        with patch("tools.oracle_zip_replay.compute_reachable_costs", fake_costs):
            apply_oracle_action_json(
                s,
                load_env,
                {awbw_pid: 0},
                envelope_awbw_player_id=awbw_pid,
            )

        apc = next(u for u in s.units[0] if u.unit_type == UnitType.APC and u.is_alive)
        self.assertEqual(len(apc.loaded_units), 1)
        self.assertEqual(int(apc.loaded_units[0].unit_id), 301)

    def test_capt_nested_move_snaps_empty_property_tail_under_truncation(self) -> None:
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
                (5, 8),
                False,
                [],
                False,
                20,
                401,
            )
        )
        s.active_player = 0
        s.action_stage = ActionStage.SELECT

        awbw_pid = 3716289
        # Neutral city (5,9) from map 77060 catalog scan; path from west.
        capt_env = {
            "action": "Capt",
            "Move": {
                "unit": {"global": _base_global(401, awbw_pid, 5, 8)},
                "paths": {"global": [{"y": 5, "x": 8}, {"y": 5, "x": 9}]},
            },
        }

        def fake_costs(_st: object, _u: object) -> dict[tuple[int, int], int]:
            return {(5, 8): 0}

        with patch("tools.oracle_zip_replay.compute_reachable_costs", fake_costs):
            apply_oracle_action_json(
                s,
                capt_env,
                {awbw_pid: 0},
                envelope_awbw_player_id=awbw_pid,
            )

        u = s.units[0][0]
        self.assertEqual((int(u.pos[0]), int(u.pos[1])), (5, 9))


if __name__ == "__main__":
    unittest.main()
