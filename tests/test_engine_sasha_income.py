"""Sasha (CO 19) "War Bonds" DTD income parity.

Engine grants +100g per income-property at the start of every turn for Sasha,
mirroring Colin's "Gold Rush" DTD bonus. Without this, mid-/late-game Sasha
treasuries drift below AWBW's by ~100g × props × turn, surfacing as the
``oracle_other`` Build no-op cluster on Sasha-vs-* games (game ``1623012``
was the first traced case).
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState


def _state_with_n_p0_cities(n: int, *, p0_co_id: int) -> GameState:
    width = 1 + n
    terrain = [[1] * width]
    properties = []
    for c in range(1, 1 + n):
        terrain[0][c] = 33  # city
        properties.append(
            PropertyState(
                terrain_id=33, row=0, col=c, owner=0, capture_points=20,
                is_hq=False, is_lab=False, is_comm_tower=False,
                is_base=False, is_airport=False, is_port=False,
            )
        )
    map_data = MapData(
        map_id=0, name="sasha-income", map_type="std",
        terrain=terrain, height=1, width=width,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=properties,
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=map_data,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(p0_co_id), make_co_state_safe(0)],
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


class TestSashaIncome(unittest.TestCase):
    def test_sasha_dtd_adds_100_per_property(self) -> None:
        st = _state_with_n_p0_cities(7, p0_co_id=19)  # Sasha
        self.assertEqual(st.funds[0], 0)
        st._grant_income(0)
        self.assertEqual(st.funds[0], 7 * (1000 + 100))

    def test_colin_dtd_unchanged(self) -> None:
        st = _state_with_n_p0_cities(7, p0_co_id=15)  # Colin
        st._grant_income(0)
        self.assertEqual(st.funds[0], 7 * (1000 + 100))

    def test_andy_dtd_no_bonus(self) -> None:
        st = _state_with_n_p0_cities(7, p0_co_id=1)  # Andy
        st._grant_income(0)
        self.assertEqual(st.funds[0], 7 * 1000)


if __name__ == "__main__":
    unittest.main()
