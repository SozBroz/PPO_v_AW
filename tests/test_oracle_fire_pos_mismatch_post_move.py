"""Regression for ``engine_pos_mismatch_post_move`` snap (oracle_fire cluster).

Two flavours, both fixed by ``_oracle_snap_mover_to_awbw_path_end``:

1. **Move terminator snap** — ``_apply_move_paths_then_terminator``: when the
   AWBW path end is unreachable in the engine (drift made the destination
   blocked / impassable / off-map for engine reachability), the engine
   truncates short and every subsequent envelope cascades into
   ``oracle_fire`` "no attacker at (R,C)". Fix: snap the engine mover to
   AWBW's intended path end after the terminator commits, when distance is
   bounded and the AWBW tile is empty (or holds the same unit).

2. **Fire-with-paths post-kill duplicate** — Fire handler:
   ``_oracle_fire_defender_row_is_postkill_noop`` early-returns when AWBW
   reposts a duplicate Fire row whose defender is already dead, but AWBW
   *still* records the attacker's post-move position in that row. Without
   snapping the mover, the engine attacker never advances to the firing
   tile, and the very next envelope that references the attacker by
   position (``Fire (no path)`` from that tile, ``Move`` continuation,
   ``Capt`` from the property) cascades. GL 1635846 day 12 j=11:
   Md.Tank id 192665470 path ``(2,10) -> (1,10) -> (1,11) -> (1,12)`` on
   a duplicate Fire — engine stayed at ``(2,10)`` and day 13 fire from
   ``(1,12)`` hit empty.
"""

from __future__ import annotations

import unittest

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType
from server.play_human import MAPS_DIR, POOL_PATH

from tools.oracle_zip_replay import (
    _ORACLE_MOVE_SNAP_MAX_TELEPORT,
    _oracle_snap_mover_to_awbw_path_end,
)


def _med_tank(player: int, pos: tuple[int, int], *, hp: int, uid: int) -> Unit:
    st = UNIT_STATS[UnitType.MED_TANK]
    return Unit(
        UnitType.MED_TANK,
        player,
        hp,
        st.max_ammo,
        st.max_fuel,
        pos,
        False,
        [],
        False,
        20,
        uid,
    )


class TestOracleFirePosMismatchPostMove(unittest.TestCase):
    """``_oracle_snap_mover_to_awbw_path_end`` contract."""

    def _state(self) -> "GameState":
        # Map 77060 is a small standard pool map used by other oracle_fire
        # tests; specific tiles are irrelevant — we only need a state with
        # an empty units table and a real ``map_data`` for occupancy checks.
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        return s

    def test_snap_to_empty_tile_succeeds(self) -> None:
        """Truncated mover at ``(2,10)`` snaps to AWBW path end ``(1,12)``."""
        s = self._state()
        u = _med_tank(0, (2, 10), hp=70, uid=61)
        s.units[0].append(u)
        ok = _oracle_snap_mover_to_awbw_path_end(s, u, (1, 12))
        self.assertTrue(ok)
        self.assertEqual(u.pos, (1, 12))
        self.assertIsNone(s.get_unit_at(2, 10))
        self.assertIs(s.get_unit_at(1, 12), u)

    def test_snap_skips_when_already_at_end(self) -> None:
        s = self._state()
        u = _med_tank(0, (1, 12), hp=70, uid=61)
        s.units[0].append(u)
        self.assertFalse(_oracle_snap_mover_to_awbw_path_end(s, u, (1, 12)))
        self.assertEqual(u.pos, (1, 12))

    def test_snap_skips_when_dead(self) -> None:
        s = self._state()
        u = _med_tank(0, (2, 10), hp=70, uid=61)
        u.hp = 0
        s.units[0].append(u)
        self.assertFalse(_oracle_snap_mover_to_awbw_path_end(s, u, (1, 12)))
        self.assertEqual(u.pos, (2, 10))

    def test_snap_skips_when_destination_occupied_by_other(self) -> None:
        """Refuse to stack: AWBW path end already holds a different unit."""
        s = self._state()
        mover = _med_tank(0, (2, 10), hp=70, uid=61)
        squatter = _med_tank(1, (1, 12), hp=100, uid=99)
        s.units[0].append(mover)
        s.units[1].append(squatter)
        self.assertFalse(
            _oracle_snap_mover_to_awbw_path_end(s, mover, (1, 12))
        )
        self.assertEqual(mover.pos, (2, 10))
        self.assertEqual(squatter.pos, (1, 12))

    def test_snap_skips_when_off_map(self) -> None:
        s = self._state()
        u = _med_tank(0, (2, 10), hp=70, uid=61)
        s.units[0].append(u)
        h = s.map_data.height
        self.assertFalse(
            _oracle_snap_mover_to_awbw_path_end(s, u, (h + 5, 0))
        )
        self.assertEqual(u.pos, (2, 10))

    def test_snap_skips_when_distance_exceeds_cap(self) -> None:
        """Refuse extreme teleports — likely a deeper bug we shouldn't mask."""
        s = self._state()
        u = _med_tank(0, (0, 0), hp=70, uid=61)
        s.units[0].append(u)
        far = (
            _ORACLE_MOVE_SNAP_MAX_TELEPORT + 1,
            _ORACLE_MOVE_SNAP_MAX_TELEPORT + 1,
        )
        self.assertFalse(_oracle_snap_mover_to_awbw_path_end(s, u, far))
        self.assertEqual(u.pos, (0, 0))


if __name__ == "__main__":
    unittest.main()
