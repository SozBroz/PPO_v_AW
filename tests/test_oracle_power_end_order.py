"""Half-turn settle before ``Power`` / ``End`` (oracle_other / day boundary)."""

from __future__ import annotations

import unittest

from engine.action import Action, ActionStage, ActionType
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS

from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import _oracle_settle_to_select_for_power


class TestOraclePowerEndOrder(unittest.TestCase):
    def test_settle_promotes_move_without_destination_to_action(self) -> None:
        """MOVE with ``selected_move_pos is None`` → full settle back to ``SELECT``."""
        m = load_map(140000, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", replay_first_mover=1)
        ap = int(s.active_player)
        u = next(
            (
                x
                for x in s.units[ap]
                if x.is_alive
                and not x.moved
                and UNIT_STATS[x.unit_type].carry_capacity == 0
            ),
            None,
        )
        if u is None:
            self.skipTest("no non-transport unmoved unit for active_player on this slice")
        s.step(Action(ActionType.SELECT_UNIT, unit_pos=u.pos))
        self.assertEqual(s.action_stage, ActionStage.MOVE)
        self.assertIsNone(s.selected_move_pos)
        _oracle_settle_to_select_for_power(s, None)
        # MOVE (no ``move_pos`` yet) → same-tile commit → ACTION → ``WAIT`` → SELECT
        # (one call mirrors site ``Power`` / ``End`` prep).
        self.assertEqual(s.action_stage, ActionStage.SELECT)
        self.assertIsNone(s.selected_unit)
        self.assertTrue(u.moved)


if __name__ == "__main__":
    unittest.main()
