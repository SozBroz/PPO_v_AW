"""Repair fallback tie-break when several Black Boats tie at the same best Manhattan (GL 1624764)."""

from __future__ import annotations

import unittest
from pathlib import Path

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.oracle_zip_replay import _oracle_fallback_repair_boat_and_ally

ROOT = Path(__file__).resolve().parents[1]


def _empty_two_player_state():
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


class TestOracleRepairDriftFallbackTiebreak(unittest.TestCase):
    def test_tiebreak_picks_lowest_boat_then_ally_ids(self) -> None:
        """GL 1624764 envelope 47: three (BB, INF) pairs at the same best_d when hp_key=10.

        ``_oracle_fallback_repair_boat_and_ally`` used to ``continue`` when
        ``acting_boat is None`` and ``len(tops) > 1``, returning ``None`` and
        breaking ``Repair`` resolution. Deterministic ``(boat_id, ally_id)``
        ordering picks the same pair as lex-first among ties.
        """
        st = _empty_two_player_state()
        eng = 1
        bb_stats = UNIT_STATS[UnitType.BLACK_BOAT]
        inf_stats = UNIT_STATS[UnitType.INFANTRY]
        boats = [
            Unit(
                unit_type=UnitType.BLACK_BOAT,
                player=eng,
                hp=40,
                ammo=bb_stats.max_ammo,
                fuel=bb_stats.max_fuel,
                pos=(10, 15),
                moved=False,
                loaded_units=[],
                is_submerged=False,
                capture_progress=20,
                unit_id=4,
            ),
            Unit(
                unit_type=UnitType.BLACK_BOAT,
                player=eng,
                hp=100,
                ammo=bb_stats.max_ammo,
                fuel=bb_stats.max_fuel,
                pos=(6, 11),
                moved=False,
                loaded_units=[],
                is_submerged=False,
                capture_progress=20,
                unit_id=5,
            ),
        ]
        allies = [
            Unit(
                unit_type=UnitType.INFANTRY,
                player=eng,
                hp=100,
                ammo=0,
                fuel=inf_stats.max_fuel,
                pos=(13, 15),
                moved=False,
                loaded_units=[],
                is_submerged=False,
                capture_progress=20,
                unit_id=127,
            ),
            Unit(
                unit_type=UnitType.INFANTRY,
                player=eng,
                hp=100,
                ammo=0,
                fuel=inf_stats.max_fuel,
                pos=(6, 8),
                moved=False,
                loaded_units=[],
                is_submerged=False,
                capture_progress=20,
                unit_id=91,
            ),
            Unit(
                unit_type=UnitType.INFANTRY,
                player=eng,
                hp=100,
                ammo=0,
                fuel=inf_stats.max_fuel,
                pos=(8, 10),
                moved=False,
                loaded_units=[],
                is_submerged=False,
                capture_progress=20,
                unit_id=119,
            ),
        ]
        st.units[eng] = boats + allies

        pair = _oracle_fallback_repair_boat_and_ally(st, eng, hp_key=10, acting_boat=None)
        self.assertIsNotNone(pair)
        assert pair is not None
        b, u = pair
        self.assertEqual(int(b.unit_id), 4)
        self.assertEqual(int(u.unit_id), 127)
        self.assertEqual(u.pos, (13, 15))


if __name__ == "__main__":
    unittest.main()
