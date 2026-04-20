"""Naval ``Lander`` does not pay idle fuel drain on a turn it actually moved.

Regression for game 1631302 (P0 Lander shuttling cargo every day): with the
universal start-of-turn idle drain the engine sank the Lander on day 19 from
fuel exhaustion, while AWBW kept it alive at fuel=3 through day 21. AWBW
empirically skips per-turn idle fuel drain on units that *moved* during their
owner's previous turn — drain only applies to units that spent the whole turn
idle.

Companion observation: when the Lander **does** sit idle a turn, AWBW *does*
drain 1 fuel, so the skip must be conditional on ``moved`` rather than blanket.
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import UNIT_STATS, Unit, UnitType


def _state_with_lander(*, lander_moved: bool, lander_fuel: int) -> GameState:
    """1x3 plain map, no properties; P0 has a Lander whose ``moved`` flag is set."""
    terrain = [[1, 1, 1]]
    map_data = MapData(
        map_id=0, name="naval-idle-drain", map_type="std",
        terrain=terrain, height=1, width=3,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=[],
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    lander_stats = UNIT_STATS[UnitType.LANDER]
    lander = Unit(
        UnitType.LANDER, 0, 100, lander_stats.max_ammo, lander_fuel,
        (0, 0), lander_moved, [], False, 0, 1,
    )
    p1_inf = UNIT_STATS[UnitType.INFANTRY]
    p1_unit = Unit(
        UnitType.INFANTRY, 1, 100, p1_inf.max_ammo, p1_inf.max_fuel,
        (0, 2), False, [], False, 0, 2,
    )
    return GameState(
        map_data=map_data,
        units={0: [lander], 1: [p1_unit]},
        funds=[0, 0],
        co_states=[make_co_state_safe(1), make_co_state_safe(1)],
        properties=[],
        turn=1,
        active_player=1,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


class TestLanderIdleDrainSkipWhenMoved(unittest.TestCase):
    def test_lander_that_moved_last_turn_does_not_idle_drain(self) -> None:
        st = _state_with_lander(lander_moved=True, lander_fuel=10)
        st._end_turn()
        self.assertEqual(st.active_player, 0)
        lander = st.units[0][0]
        self.assertEqual(
            lander.fuel, 10,
            f"Lander that moved last turn must NOT lose idle fuel; got {lander.fuel}",
        )
        self.assertFalse(lander.moved, "moved flag must be reset for new turn")

    def test_lander_that_idled_last_turn_loses_one_fuel(self) -> None:
        st = _state_with_lander(lander_moved=False, lander_fuel=10)
        st._end_turn()
        self.assertEqual(st.active_player, 0)
        lander = st.units[0][0]
        self.assertEqual(
            lander.fuel, 9,
            f"Lander that idled last turn must lose 1 idle fuel; got {lander.fuel}",
        )

    def test_idle_lander_at_fuel_one_drains_to_zero_and_sinks(self) -> None:
        st = _state_with_lander(lander_moved=False, lander_fuel=1)
        st._end_turn()
        self.assertEqual(
            st.units[0], [],
            "Idle Lander draining to fuel=0 on open sea must sink (no port refuel)",
        )

    def test_moved_lander_at_fuel_one_survives(self) -> None:
        st = _state_with_lander(lander_moved=True, lander_fuel=1)
        st._end_turn()
        self.assertEqual(len(st.units[0]), 1)
        self.assertEqual(st.units[0][0].fuel, 1)


if __name__ == "__main__":
    unittest.main()
