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
    _oracle_get_killed_awbw_ids,
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
        """GL **1631194**: hp 0, defender id gone, unrelated unit on recorded tile.

        The original gate (``_unit_by_awbw_units_id is None``) was uninformative
        in the zip-replay lane (engine units don't carry AWBW ids), so the skip
        is now keyed on the per-state set of AWBW ids the oracle has already
        applied as killed by a prior ``Fire`` row. We seed that set here to
        model "earlier envelope already killed defender 192332445; this row is
        the duplicate re-emit".
        """
        s = self._empty_state()
        r, c = 9, 4
        s.units[1].append(_t_copter(1, (r, c), uid=1001))
        defender = {
            "units_y": r,
            "units_x": c,
            "units_hit_points": 0,
            "units_id": 192332445,
        }
        _oracle_get_killed_awbw_ids(s).add(192332445)
        self.assertTrue(
            _oracle_fire_no_path_postkill_dead_defender_orphan_tile_reoccupied(s, defender)
        )

    def test_dead_defender_first_strike_is_not_noop(self) -> None:
        """GL **1628985**: first ``hp=0`` row IS the killing strike; do not skip.

        Engine has the original defender alive on the recorded tile; the
        oracle has not yet applied any strike on this AWBW id. The function
        must return False so the caller proceeds to apply the kill (rather
        than orphaning the engine unit and corrupting the state into a stack
        on the next envelope).
        """
        s = self._empty_state()
        r, c = 10, 9
        s.units[1].append(_t_copter(1, (r, c), uid=7))
        defender = {
            "units_y": r,
            "units_x": c,
            "units_hit_points": 0,
            "units_id": 192111511,
        }
        self.assertFalse(
            _oracle_fire_no_path_postkill_dead_defender_orphan_tile_reoccupied(s, defender)
        )

    def test_low_hp_orphan_vs_air_when_get_base_damage_none(self) -> None:
        """Regression: orphan air defender hp 1-2 + live unrelated unit on tile -> skip predicate true.

        Originally gated on ``get_base_damage(INFANTRY, B_COPTER) is None``; agent4
        filled that chart cell (see ``data/damage_table.json``) so the gate is now
        class-based on the orphan-tile occupant. Test name retained for blame
        history; behaviour is unchanged for this fixture (Inf vs B-Copter, hp 1).
        """
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
