"""Oracle ``_oracle_advance_turn_until_player`` must call ``_end_turn`` even
when the engine refuses ``END_TURN`` due to unmoved units.

Regression for the ``oracle_other`` Build no-op cluster
("insufficient funds (need 1000$, have 0$)") seen in game ``1618984`` and
~30 sibling Andy/Andy mirrors. AWBW lets the active player end their turn
even with unmoved units; the engine gates ``END_TURN`` in
``get_legal_actions`` for RL training only. Before the fix, the oracle's
seat-snap fallback bypassed ``GameState._end_turn`` entirely, so the
opponent never received start-of-turn income / fuel drain / resupply.
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.oracle_zip_replay import _oracle_advance_turn_until_player


def _two_player_state_with_unmoved_p1_unit(*, p0_props: int = 5) -> GameState:
    """Tiny map: row of plains + ``p0_props`` neutral cities; P1 has 1 unmoved infantry."""
    width = 1 + p0_props
    terrain = [[1] * width]
    properties = []
    for c in range(1, 1 + p0_props):
        terrain[0][c] = 33  # city tile id
        properties.append(
            PropertyState(
                terrain_id=33, row=0, col=c, owner=0, capture_points=20,
                is_hq=False, is_lab=False, is_comm_tower=False,
                is_base=False, is_airport=False, is_port=False,
            )
        )
    map_data = MapData(
        map_id=0, name="oracle-end-turn-income", map_type="std",
        terrain=terrain, height=1, width=width,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=properties,
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    inf = UNIT_STATS[UnitType.INFANTRY]
    p1_unit = Unit(
        UnitType.INFANTRY, 1, 100, inf.max_ammo, inf.max_fuel,
        (0, 0), False, [], False, 20, 1,
    )
    return GameState(
        map_data=map_data,
        units={0: [], 1: [p1_unit]},
        funds=[0, 0],
        co_states=[make_co_state_safe(1), make_co_state_safe(1)],
        properties=map_data.properties,
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


class TestOracleAdvanceTurnGrantsIncome(unittest.TestCase):
    def test_force_end_turn_grants_opponent_income_when_p1_has_unmoved_units(self) -> None:
        st = _two_player_state_with_unmoved_p1_unit(p0_props=5)
        self.assertEqual(st.active_player, 1)
        self.assertEqual(st.funds[0], 0)

        # Sanity: get_legal_actions refuses END_TURN because P1 has an unmoved unit.
        from engine.action import ActionType, get_legal_actions
        legal = get_legal_actions(st)
        self.assertFalse(any(a.action_type == ActionType.END_TURN for a in legal))

        _oracle_advance_turn_until_player(st, want_eng=0, before_engine_step=None)

        self.assertEqual(st.active_player, 0)
        # 5 owned cities × 1000g = 5000g income for P0.
        self.assertEqual(st.funds[0], 5000)
        # P1's unmoved unit must have moved=False reset for the new turn boundary
        # (sanity that the engine actually ran _end_turn rather than the bare snap).
        self.assertFalse(st.units[1][0].moved)


if __name__ == "__main__":
    unittest.main()
