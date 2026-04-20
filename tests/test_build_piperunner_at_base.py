"""Piperunner is producible from any owned Base unless the map bans it.

Regression for `oracle_build` cluster (PIPERUNNER not producible) in games
1631113 and 1632872 — both maps have neutral `unit_bans` (no Piperunner) and
emit `Build Piperunner` on a regular friendly Base.
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage, get_producible_units
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.terrain import get_terrain
from engine.unit import UnitType


def _state_owned_base_factory(*, funds0: int, unit_bans: list[str]) -> GameState:
    """1×2 map: plain at (0,0), P0 base at (0,1)."""
    terrain = [[1, 39]]
    prop = PropertyState(
        terrain_id=39,
        row=0,
        col=1,
        owner=0,
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
        name="piperunner-build",
        map_type="std",
        terrain=terrain,
        height=1,
        width=2,
        cap_limit=99,
        unit_limit=50,
        unit_bans=list(unit_bans),
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
        funds=[funds0, 10_000],
        co_states=[make_co_state_safe(0), make_co_state_safe(0)],
        properties=map_data.properties,
        turn=1,
        active_player=0,
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


class TestPiperunnerProducibleAtBase(unittest.TestCase):
    def test_base_lists_piperunner_when_unbanned(self) -> None:
        info = get_terrain(35)
        produced = set(get_producible_units(info, []))
        self.assertIn(
            UnitType.PIPERUNNER,
            produced,
            "AWBW: any owned base should be able to build Piperunner unless banned",
        )

    def test_base_excludes_piperunner_when_banned(self) -> None:
        info = get_terrain(35)
        produced = set(get_producible_units(info, ["Piperunner"]))
        self.assertNotIn(
            UnitType.PIPERUNNER,
            produced,
            "Piperunner ban must remove it from base producibles",
        )

    def test_airport_does_not_produce_piperunner(self) -> None:
        info = get_terrain(36)
        produced = set(get_producible_units(info, []))
        self.assertNotIn(
            UnitType.PIPERUNNER,
            produced,
            "Piperunner is a ground unit; airports must not produce it",
        )

    def test_port_does_not_produce_piperunner(self) -> None:
        info = get_terrain(37)
        produced = set(get_producible_units(info, []))
        self.assertNotIn(
            UnitType.PIPERUNNER,
            produced,
            "Piperunner is a ground unit; ports must not produce it",
        )

    def test_engine_apply_build_accepts_piperunner_on_base(self) -> None:
        from engine.action import Action, ActionType

        st = _state_owned_base_factory(funds0=20_000, unit_bans=[])
        action = Action(
            ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.PIPERUNNER
        )
        st.step(action)
        self.assertEqual(
            len(st.units[0]),
            1,
            "Piperunner should materialize on an owned base with funds",
        )
        self.assertEqual(st.units[0][0].unit_type, UnitType.PIPERUNNER)
        self.assertEqual(st.funds[0], 0, "20000$ Piperunner cost must debit funds")


if __name__ == "__main__":
    unittest.main()
