"""Naval build sanity tests.

AWBW only allows naval units (Black Boat, Lander, Sub, Cruiser, Battleship,
Carrier, Gunboat) to be produced on **port** tiles. `get_producible_units`
enforces this directly by tile type. `_apply_build` also rejects BUILD on
non-factory terrain. These tests lock both in place so future refactors
cannot silently regress the rule exposed by replay 135015.
"""
from __future__ import annotations

import unittest

from engine.action import Action, ActionType, ActionStage, get_producible_units
from engine.co import make_co_state_safe
from engine.game import GameState, IllegalActionError
from engine.map_loader import MapData, PropertyState
from engine.terrain import get_terrain
from engine.unit import UnitType


NAVAL_TYPES = {
    UnitType.BLACK_BOAT, UnitType.LANDER, UnitType.SUBMARINE,
    UnitType.CRUISER, UnitType.BATTLESHIP, UnitType.CARRIER, UnitType.GUNBOAT,
}


def _state_with_factory(*, terrain_id: int, is_base: bool, is_airport: bool, is_port: bool) -> GameState:
    """1x2 map with a single owned factory tile of the given type."""
    terrain = [[1, terrain_id]]
    prop = PropertyState(
        terrain_id=terrain_id,
        row=0,
        col=1,
        owner=0,
        capture_points=20,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=is_base,
        is_airport=is_airport,
        is_port=is_port,
    )
    map_data = MapData(
        map_id=0,
        name="naval-guard-test",
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
        funds=[99_000, 99_000],
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


class TestNavalBuildTerrain(unittest.TestCase):
    def test_base_produces_no_naval(self) -> None:
        info = get_terrain(35)  # neutral base
        produced = set(get_producible_units(info, []))
        self.assertTrue(produced.isdisjoint(NAVAL_TYPES),
                        f"base must not list naval units; got {produced & NAVAL_TYPES}")

    def test_airport_produces_no_naval(self) -> None:
        info = get_terrain(36)  # neutral airport
        produced = set(get_producible_units(info, []))
        self.assertTrue(produced.isdisjoint(NAVAL_TYPES),
                        f"airport must not list naval units; got {produced & NAVAL_TYPES}")

    def test_port_produces_naval(self) -> None:
        info = get_terrain(37)  # neutral port
        produced = set(get_producible_units(info, []))
        self.assertIn(UnitType.BLACK_BOAT, produced,
                      "port must be able to produce Black Boat")
        self.assertIn(UnitType.LANDER, produced)

    def test_crafted_black_boat_on_base_rejected(self) -> None:
        """Phase 10M: crafted illegal naval BUILD hits STEP-GATE before `_apply_build`."""
        state = _state_with_factory(terrain_id=35, is_base=True, is_airport=False, is_port=False)
        funds_before = state.funds[0]
        action = Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.BLACK_BOAT)

        with self.assertRaises(IllegalActionError):
            state.step(action)

        self.assertEqual(len(state.units[0]), 0, "Black Boat must not materialize on a base")
        self.assertEqual(state.funds[0], funds_before, "funds must not be debited on rejection")

    def test_crafted_black_boat_on_port_succeeds(self) -> None:
        state = _state_with_factory(terrain_id=37, is_base=False, is_airport=False, is_port=True)
        action = Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.BLACK_BOAT)

        state.step(action)

        self.assertEqual(len(state.units[0]), 1, "Black Boat should build on a port")
        self.assertEqual(state.units[0][0].unit_type, UnitType.BLACK_BOAT)
        self.assertGreater(state.units[0][0].unit_id, 0,
                           "newly built unit must receive a stable unit_id")


if __name__ == "__main__":
    unittest.main()
