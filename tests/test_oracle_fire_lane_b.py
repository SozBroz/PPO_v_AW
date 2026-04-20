"""Lane B (GL) ``oracle_fire`` duplicate / post-kill rows (``tools/oracle_zip_replay``)."""

from __future__ import annotations

import unittest

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType
from server.play_human import MAPS_DIR, POOL_PATH

from tools.oracle_zip_replay import (
    _oracle_fire_defender_row_is_postkill_noop,
    _oracle_fire_no_path_low_hp_orphan_unmodelled_vs_air,
    _oracle_fire_no_path_postkill_dead_defender_orphan_tile_reoccupied,
)


def _t_copter(p: int, pos: tuple[int, int], *, uid: int) -> Unit:
    st = UNIT_STATS[UnitType.T_COPTER]
    return Unit(
        UnitType.T_COPTER,
        p,
        100,
        st.max_ammo,
        st.max_fuel,
        pos,
        False,
        [],
        False,
        20,
        uid,
    )


class TestOracleFireDefenderPostkillNoopLaneB(unittest.TestCase):
    """``_oracle_fire_defender_row_is_postkill_noop`` — lane B duplicate ``Fire`` rows."""

    def _empty_state(self):
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        return s

    def test_postkill_empty_tile_is_noop(self) -> None:
        """Classic post-kill duplicate: hp<=0 and defender tile empty."""
        s = self._empty_state()
        defender = {
            "units_y": 5,
            "units_x": 5,
            "units_hit_points": 0,
            "units_id": 1,
        }
        self.assertTrue(_oracle_fire_defender_row_is_postkill_noop(s, defender))

    def test_dead_defender_orphan_reoccupied_tile_is_noop(self) -> None:
        """GL **1631194**: hp 0, defender id gone, unrelated unit on recorded tile."""
        s = self._empty_state()
        r, c = 9, 4
        s.units[1].append(_t_copter(1, (r, c), uid=1001))
        defender = {
            "units_y": r,
            "units_x": c,
            "units_hit_points": 0,
            "units_id": 192332445,
        }
        self.assertTrue(
            _oracle_fire_no_path_postkill_dead_defender_orphan_tile_reoccupied(s, defender)
        )

    def test_low_hp_orphan_vs_air_when_get_base_damage_none(self) -> None:
        """Regression: INFANTRY vs B-COPTER returns ``None`` from chart — skip predicate true."""
        s = self._empty_state()
        bc_st = UNIT_STATS[UnitType.B_COPTER]
        bc = Unit(
            UnitType.B_COPTER,
            1,
            80,
            bc_st.max_ammo,
            bc_st.max_fuel,
            (10, 11),
            False,
            [],
            False,
            20,
            802,
        )
        s.units[1].append(bc)
        inf_st = UNIT_STATS[UnitType.INFANTRY]
        inf = Unit(
            UnitType.INFANTRY,
            0,
            100,
            inf_st.max_ammo,
            inf_st.max_fuel,
            (10, 10),
            False,
            [],
            False,
            20,
            801,
        )
        s.units[0].append(inf)
        defender = {
            "units_y": 10,
            "units_x": 11,
            "units_hit_points": 1,
            "units_id": 9919991,
        }
        self.assertTrue(
            _oracle_fire_no_path_low_hp_orphan_unmodelled_vs_air(
                s, defender, 10, 10, 10, 11
            )
        )


if __name__ == "__main__":
    unittest.main()
