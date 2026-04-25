"""``units_hit_points`` post-heal hint must not match display ``want+1`` allies (gid 1624307)."""

from __future__ import annotations

import unittest
from pathlib import Path

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.oracle_zip_replay import _resolve_repair_target_tile

ROOT = Path(__file__).resolve().parents[1]


class TestOracleRepairTargetPreHealHint(unittest.TestCase):
    def test_post_heal_hint_9_picks_display_8_ally_not_full_hp_neighbor(self) -> None:
        """PHP ``repaired.global.units_hit_points`` is post-heal (+1 bar).

        A ±1 fuzzy filter admitted display 10 (full HP) as well as display 8
        when ``want=9``, so multi–Black-Boat boards picked the wrong orth
        neighbour and ``Repair`` became a no-heal (engine richer than PHP).
        """
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
        eng = 0
        bb_stats = UNIT_STATS[UnitType.BLACK_BOAT]
        tank_stats = UNIT_STATS[UnitType.TANK]
        bb_a = Unit(
            UnitType.BLACK_BOAT,
            eng,
            100,
            bb_stats.max_ammo,
            bb_stats.max_fuel,
            (12, 9),
            False,
            [],
            False,
            20,
            1,
        )
        wounded = Unit(
            UnitType.TANK,
            eng,
            73,
            tank_stats.max_ammo,
            tank_stats.max_fuel,
            (11, 9),
            False,
            [],
            False,
            20,
            2,
        )
        bb_b = Unit(
            UnitType.BLACK_BOAT,
            eng,
            100,
            bb_stats.max_ammo,
            bb_stats.max_fuel,
            (5, 17),
            False,
            [],
            False,
            20,
            3,
        )
        full = Unit(
            UnitType.TANK,
            eng,
            100,
            tank_stats.max_ammo,
            tank_stats.max_fuel,
            (6, 17),
            False,
            [],
            False,
            20,
            4,
        )
        st.units[eng] = [bb_a, wounded, bb_b, full]
        repair_block = {
            "repaired": {
                "global": {
                    "units_id": 999888777,
                    "units_hit_points": 9,
                }
            }
        }
        r, c = _resolve_repair_target_tile(
            st,
            repair_block,
            eng=eng,
            boat_hint=None,
            envelope_awbw_player_id=None,
        )
        self.assertEqual((r, c), (11, 9))


if __name__ == "__main__":
    unittest.main()
