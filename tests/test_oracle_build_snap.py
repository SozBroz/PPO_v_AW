"""Neutral production ownership snap for oracle ``Build`` (``oracle_other`` / economy)."""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.terrain import get_terrain
from engine.unit import UnitType

from tools.oracle_zip_replay import _oracle_snap_neutral_production_owner_for_build


def _state_neutral_base_factory(*, funds0: int) -> GameState:
    """1×2 map: plain at (0,0), neutral base at (0,1) — matches ``test_build_guard`` layout."""
    terrain = [[1, 35]]
    prop = PropertyState(
        terrain_id=35,
        row=0,
        col=1,
        owner=None,
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
        name="oracle-build-snap",
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


class TestOracleBuildNeutralSnap(unittest.TestCase):
    def test_snap_sets_owner_when_neutral_base_empty_and_affordable(self) -> None:
        st = _state_neutral_base_factory(funds0=10_000)
        tid = st.map_data.terrain[0][1]
        self.assertTrue(get_terrain(tid).is_base)
        p0 = st.get_property_at(0, 1)
        assert p0 is not None
        self.assertIsNone(p0.owner)
        ok = _oracle_snap_neutral_production_owner_for_build(
            st, 0, 1, 0, UnitType.INFANTRY
        )
        self.assertTrue(ok)
        p1 = st.get_property_at(0, 1)
        self.assertIsNotNone(p1)
        assert p1 is not None
        self.assertEqual(int(p1.owner), 0)

    def test_snap_no_op_when_factory_occupied(self) -> None:
        from engine.unit import UNIT_STATS, Unit

        st = _state_neutral_base_factory(funds0=10_000)
        inf = UNIT_STATS[UnitType.INFANTRY]
        st.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                100,
                inf.max_ammo,
                inf.max_fuel,
                (0, 1),
                True,
                [],
                False,
                20,
                1,
            )
        )
        ok = _oracle_snap_neutral_production_owner_for_build(
            st, 0, 1, 0, UnitType.INFANTRY
        )
        self.assertFalse(ok)
        self.assertIsNone(st.get_property_at(0, 1).owner)


if __name__ == "__main__":
    unittest.main()
