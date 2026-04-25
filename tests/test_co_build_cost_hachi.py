"""Hachi (CO 17) build-cost canon parity.

Source: AWBW CO Chart https://awbw.amarriner.com/co.php — "Units cost 10% less"
(D2D, applies to **every build site**, not just bases).

Phase 11A replaced the old "50% on ``terrain.is_base`` only" heuristic in
``engine.action._build_cost`` with the chart rule (``×0.9`` on every build).
Tank list cost 7000 → Hachi pays 6300 anywhere a build is legal.
See ``docs/oracle_exception_audit/phase11a_kindle_hachi_canon.md`` for the
audit and rollback record.
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage, _build_cost
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import UNIT_STATS, UnitType


# ---- terrain ids (engine/terrain.py) -------------------------------------
NEUTRAL_BASE    = 35
NEUTRAL_AIRPORT = 36
NEUTRAL_PORT    = 37


def _state(co_id: int, terrain_id: int) -> tuple[GameState, tuple[int, int]]:
    """Return a 1x1 fixture state where the only tile is ``terrain_id``.

    The build position is (0, 0); ``_build_cost`` is independent of legality
    machinery — it only reads the CO and the terrain at ``pos``.
    """
    terrain = [[terrain_id]]
    properties = [PropertyState(
        terrain_id=terrain_id, row=0, col=0, owner=0, capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False,
        is_base=(terrain_id == NEUTRAL_BASE),
        is_airport=(terrain_id == NEUTRAL_AIRPORT),
        is_port=(terrain_id == NEUTRAL_PORT),
    )]
    map_data = MapData(
        map_id=0, name="hachi-build", map_type="std",
        terrain=terrain, height=1, width=1,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=properties,
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    state = GameState(
        map_data=map_data,
        units={0: [], 1: []},
        funds=[100_000, 100_000],
        co_states=[make_co_state_safe(co_id), make_co_state_safe(0)],
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
    return state, (0, 0)


class TestHachiBuildCost(unittest.TestCase):
    """AWBW CO Chart: Hachi units cost 10% less on every build (×0.9)."""

    def test_hachi_tank_on_base(self) -> None:
        state, pos = _state(co_id=17, terrain_id=NEUTRAL_BASE)
        cost = _build_cost(UnitType.TANK, state, player=0, pos=pos)
        self.assertEqual(
            cost, 6300,
            "Hachi Tank on a base must be 90% of 7000 (was 3500 under old "
            "50%-on-base heuristic).",
        )

    def test_hachi_b_copter_on_airport(self) -> None:
        # Pre-Phase-11A: airport build returned full cost (no 50% discount).
        # New chart rule: every build site gets the 10%-off discount.
        state, pos = _state(co_id=17, terrain_id=NEUTRAL_AIRPORT)
        list_cost = UNIT_STATS[UnitType.B_COPTER].cost
        cost = _build_cost(UnitType.B_COPTER, state, player=0, pos=pos)
        expected = int(list_cost * 0.9)
        self.assertEqual(
            cost, expected,
            f"Hachi B-Copter on an airport must be 90% of {list_cost} "
            f"(={expected}); pre-Phase-11A returned full {list_cost}.",
        )

    def test_hachi_lander_on_port(self) -> None:
        state, pos = _state(co_id=17, terrain_id=NEUTRAL_PORT)
        list_cost = UNIT_STATS[UnitType.LANDER].cost
        cost = _build_cost(UnitType.LANDER, state, player=0, pos=pos)
        expected = int(list_cost * 0.9)
        self.assertEqual(cost, expected)

    def test_non_hachi_tank_unchanged(self) -> None:
        # Andy (1) has no build-cost modifier.
        state, pos = _state(co_id=1, terrain_id=NEUTRAL_BASE)
        cost = _build_cost(UnitType.TANK, state, player=0, pos=pos)
        self.assertEqual(cost, 7000)

    def test_colin_still_80_percent(self) -> None:
        # Sanity: Colin's 80% rule remains intact alongside Hachi's new 90%.
        state, pos = _state(co_id=15, terrain_id=NEUTRAL_BASE)
        cost = _build_cost(UnitType.TANK, state, player=0, pos=pos)
        self.assertEqual(cost, 5600)


if __name__ == "__main__":
    unittest.main()
