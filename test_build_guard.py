"""Defense-in-depth tests for `_apply_build`.

`get_legal_actions` already restricts BUILD to factories owned by the
active player, but engines are also callable with hand-constructed
`Action` objects. These tests bypass the legal-action filter and feed
`GameState.step` a crafted BUILD, then verify that the guard in
`_apply_build` rejects the move without mutating state.
"""
from __future__ import annotations

import unittest

from engine.action import Action, ActionType, ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import UnitType


def _minimal_state(*, active_player: int, factory_owner) -> GameState:
    """Build a 1x2 map with a single Base tile whose ownership is configurable."""
    terrain = [[1, 35]]  # plain, neutral base; ownership is carried by PropertyState
    prop = PropertyState(
        terrain_id=35,
        row=0,
        col=1,
        owner=factory_owner,
        capture_points=20,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=True,
        is_airport=False,
        is_port=False,
    )
    map_data = MapData(
        map_id=0,
        name="guard-test",
        map_type="std",
        terrain=terrain,
        height=1,
        width=2,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[prop],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=map_data,
        units={0: [], 1: []},
        funds=[10_000, 10_000],
        co_states=[make_co_state_safe(0), make_co_state_safe(0)],
        properties=map_data.properties,
        turn=1,
        active_player=active_player,
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


class TestBuildGuard(unittest.TestCase):
    def test_build_on_own_factory_succeeds(self) -> None:
        state = _minimal_state(active_player=0, factory_owner=0)
        action = Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.INFANTRY)

        state.step(action)

        self.assertEqual(len(state.units[0]), 1, "P0 should own the new infantry")
        self.assertEqual(state.units[0][0].player, 0)
        self.assertLess(state.funds[0], 10_000, "funds should be debited")
        self.assertEqual(len(state.units[1]), 0, "P1 must not receive a unit")

    def test_build_on_opponent_factory_rejected(self) -> None:
        # Active player is 1, factory is owned by 0. Crafted action must be rejected.
        state = _minimal_state(active_player=1, factory_owner=0)
        action = Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.INFANTRY)

        state.step(action)

        self.assertEqual(len(state.units[0]), 0, "no unit may be placed for P0")
        self.assertEqual(len(state.units[1]), 0, "no unit may be placed for P1 either")
        self.assertEqual(state.funds[1], 10_000, "attacker funds must not be debited")
        self.assertEqual(state.funds[0], 10_000, "defender funds must not be debited")

    def test_build_on_neutral_factory_rejected(self) -> None:
        state = _minimal_state(active_player=0, factory_owner=None)
        action = Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.INFANTRY)

        state.step(action)

        self.assertEqual(len(state.units[0]), 0)
        self.assertEqual(state.funds[0], 10_000)


if __name__ == "__main__":
    unittest.main()
