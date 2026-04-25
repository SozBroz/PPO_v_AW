"""Pre-envelope PHP ``units[].id`` → tile cache lets ``Delete`` scrap predeploy blockers.

Engine ``unit_id`` is monotonic at spawn and does not match AWBW database ids on
frame0. ``_unit_by_awbw_units_id`` therefore misses; the tile from the PHP
snapshot for that id must disambiguate (GL gid 1628198).
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import apply_oracle_action_json, oracle_set_php_id_tile_cache


class TestOracleDeletePhpIdTileFallback(unittest.TestCase):
    def test_delete_kills_occupier_when_awbw_id_not_engine_unit_id(self) -> None:
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        ist = UNIT_STATS[UnitType.INFANTRY]
        awbw_db_id = 192112547
        engine_only_id = 1
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
                engine_only_id,
            )
        )
        s.active_player = 0
        s.action_stage = ActionStage.SELECT
        awbw_pid = 90001
        frame = {
            "units": {
                "u1": {
                    "id": awbw_db_id,
                    "y": 5,
                    "x": 5,
                    "carried": "N",
                }
            }
        }
        oracle_set_php_id_tile_cache(s, frame)
        obj = {
            "action": "Delete",
            "Delete": {"unitId": {"global": awbw_db_id}},
        }
        apply_oracle_action_json(
            s,
            obj,
            {awbw_pid: 0},
            envelope_awbw_player_id=awbw_pid,
        )
        self.assertIsNone(s.get_unit_at(5, 5))
        self.assertEqual(sum(1 for u in s.units[0] if u.is_alive), 0)


if __name__ == "__main__":
    unittest.main()
