"""Phase 11J-F2-KOAL-FU-ORACLE — Capt no-path branch must not flip to defender's seat
when the engine's active player already matches the envelope.

Smoking gun (gid 1630794, env 37, day 19, Koal P1):

    [0] Power → cop_active = True
    [1] Fire  (cop_active still True)
    [2] Capt — buildingInfo carries `buildings_team='3768109'` (the *defender* /
              opponent AWBW id, not the capturer). Pre-fix, the Capt branch in
              ``tools/oracle_zip_replay.py::_apply_oracle_action_json_body``
              passed that id to ``_oracle_ensure_envelope_seat``, which called
              ``_oracle_advance_turn_until_player`` → ``END_TURN`` →
              ``_end_turn`` → cleared ``cop_active``.
    ...
    [18] Load — Infantry (2,7) → (1,10) requires Koal COP +1 movement; with
              cop_active cleared, the move is unreachable → ValueError.

The fix: when ``state.active_player == awbw_to_engine[envelope_awbw_player_id]``
already, prefer the envelope's seat (no-op) over ``buildings_players_id`` /
``buildings_team`` (the defender on a contested capture).

This module pins the seat-preservation contract:

  Test 1: Koal cop_active=True, captures opponent property → cop_active still True.
  Test 2: Andy capturing a *neutral* property — no envelope-vs-defender conflict;
          behavior unchanged from the pre-patch baseline (no spurious END_TURN).
  Test 3: Andy capturing opponent property — even with no movement-bonus state to
          lose, no spurious END_TURN fires (active_player and turn_idx invariant).
  Test 4: Sami scop_active=True, captures opponent property → scop_active still True.
"""

from __future__ import annotations

import unittest
from typing import Optional

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.oracle_zip_replay import apply_oracle_action_json


# AWBW terrain IDs:
PLAIN = 1
NEUTRAL_CITY = 34
ORANGE_STAR_CITY = 38
BLUE_MOON_CITY = 43


# AWBW player IDs (arbitrary integers, mirror PHP exports). engine seat is the
# value; the AWBW id is the key in awbw_to_engine.
AWBW_P0 = 1_000_000
AWBW_P1 = 2_000_000


def _row(width: int, override: Optional[dict[int, int]] = None) -> list[int]:
    row = [PLAIN] * width
    if override:
        for c, tid in override.items():
            row[c] = tid
    return row


def _build_state(
    *,
    co_p0_id: int,
    co_p1_id: int,
    width: int,
    capturer_player: int,
    capturer_pos: tuple[int, int],
    capturer_type: UnitType,
    property_pos: tuple[int, int],
    property_terrain_id: int,
    property_owner: Optional[int],
    capturer_capture_progress: int = 20,
    active_player: int = 0,
    cop_active_p0: bool = False,
    cop_active_p1: bool = False,
    scop_active_p0: bool = False,
    scop_active_p1: bool = False,
) -> GameState:
    pr, pc = property_pos
    terrain_row = _row(width, {pc: property_terrain_id})
    properties = [
        PropertyState(
            terrain_id=property_terrain_id, row=pr, col=pc, owner=property_owner,
            capture_points=20,
            is_hq=False, is_lab=False, is_comm_tower=False,
            is_base=False, is_airport=False, is_port=False,
        )
    ]
    map_data = MapData(
        map_id=0, name="oracle-capt-envelope-seat", map_type="std",
        terrain=[terrain_row], height=1, width=width,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=properties,
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    stats = UNIT_STATS[capturer_type]
    capturer = Unit(
        capturer_type, capturer_player, 100, stats.max_ammo, stats.max_fuel,
        capturer_pos, False, [], False, capturer_capture_progress, 1,
    )
    units = {0: [], 1: []}
    units[capturer_player] = [capturer]
    co_states = [make_co_state_safe(co_p0_id), make_co_state_safe(co_p1_id)]
    co_states[0].cop_active = cop_active_p0
    co_states[0].scop_active = scop_active_p0
    co_states[1].cop_active = cop_active_p1
    co_states[1].scop_active = scop_active_p1
    return GameState(
        map_data=map_data,
        units=units,
        funds=[0, 0],
        co_states=co_states,
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


def _capt_action(
    building_row: int,
    building_col: int,
    *,
    buildings_team: Optional[str] = None,
    buildings_players_id: Optional[int] = None,
    buildings_capture: int = 6,
) -> dict:
    bi: dict = {
        "buildings_capture": buildings_capture,
        "buildings_id": 999_111_222,
        "buildings_x": building_col,
        "buildings_y": building_row,
    }
    if buildings_team is not None:
        bi["buildings_team"] = buildings_team
    if buildings_players_id is not None:
        bi["buildings_players_id"] = buildings_players_id
    return {
        "action": "Capt",
        "Move": [],
        "Capt": {
            "action": "Capt",
            "buildingInfo": bi,
            "vision": {"global": {"onCapture": {"x": building_col, "y": building_row}}},
            "income": None,
        },
    }


class TestOracleCaptEnvelopeSeatPreservesCopState(unittest.TestCase):
    """Capt no-path must not invent an END_TURN when ``active_player`` already
    matches the envelope's ``p:`` seat."""

    def test_koal_cop_capt_contested_property_preserves_cop_active(self) -> None:
        """gid 1630794 mirror — Koal P1 with COP active captures a P0-owned city.

        Pre-fix: ``buildings_team='AWBW_P0'`` → seat-flip to engine 0 → END_TURN
        → ``cop_active`` cleared. Post-fix: envelope already aligned (active_player==1
        matches awbw_to_engine[AWBW_P1]) → no flip → ``cop_active`` preserved.
        """
        st = _build_state(
            co_p0_id=14, co_p1_id=21,                  # Jess vs Koal
            width=4,
            capturer_player=1, capturer_pos=(0, 1), capturer_type=UnitType.INFANTRY,
            property_pos=(0, 2), property_terrain_id=ORANGE_STAR_CITY,
            property_owner=0,                           # P0 (the defender / opponent)
            active_player=1,                            # Koal's half-turn
            cop_active_p1=True,                         # Forced March on
        )
        awbw_to_engine = {AWBW_P0: 0, AWBW_P1: 1}
        action = _capt_action(
            building_row=0, building_col=2,
            buildings_team=str(AWBW_P0),                # defender id (opponent)
        )
        apply_oracle_action_json(
            st, action, awbw_to_engine,
            envelope_awbw_player_id=AWBW_P1,            # capturer's envelope
        )
        self.assertEqual(st.active_player, 1,
                         "Koal must remain the active player after Capt")
        self.assertTrue(st.co_states[1].cop_active,
                        "Koal cop_active must survive the Capt envelope "
                        "(no spurious END_TURN to defender's seat)")
        self.assertFalse(st.co_states[0].cop_active)
        self.assertFalse(st.co_states[0].scop_active)

    def test_andy_capt_neutral_property_no_behavior_change(self) -> None:
        """Neutral capture: no defender, no contested-capture conflict.

        ``buildings_team`` and ``buildings_players_id`` both absent → no risk of
        seat-flip to opponent. Verifies the new short-circuit path doesn't break
        the existing fall-through to ``envelope_awbw_player_id``.
        """
        st = _build_state(
            co_p0_id=1, co_p1_id=1,                    # Andy vs Andy
            width=4,
            capturer_player=0, capturer_pos=(0, 1), capturer_type=UnitType.INFANTRY,
            property_pos=(0, 2), property_terrain_id=NEUTRAL_CITY,
            property_owner=None,
            active_player=0,
        )
        awbw_to_engine = {AWBW_P0: 0, AWBW_P1: 1}
        action = _capt_action(building_row=0, building_col=2)  # no team / pid
        apply_oracle_action_json(
            st, action, awbw_to_engine,
            envelope_awbw_player_id=AWBW_P0,
        )
        self.assertEqual(st.active_player, 0)

    def test_andy_capt_opponent_property_no_spurious_end_turn(self) -> None:
        """Andy (no movement-bonus power) capturing opponent property.

        Andy COP / SCOP don't carry a +N movement bonus that would trip the
        smoking-gun, but the seat-flip-to-defender path was still wrong: it
        consumed an END_TURN, advancing the turn counter, draining idle fuel,
        and generally desynchronizing the half-turn from AWBW. This test pins
        that NO turn flip happens when the envelope already aligns.
        """
        # Track _end_turn calls during application.
        end_turn_calls = {"n": 0}
        st = _build_state(
            co_p0_id=1, co_p1_id=1,
            width=4,
            capturer_player=0, capturer_pos=(0, 1), capturer_type=UnitType.INFANTRY,
            property_pos=(0, 2), property_terrain_id=BLUE_MOON_CITY,
            property_owner=1,                           # P1 (defender / opponent)
            active_player=0,                            # P0 capturing
        )
        from engine.game import GameState as _GS
        orig_end_turn = _GS._end_turn
        try:
            def _counted(self):
                end_turn_calls["n"] += 1
                return orig_end_turn(self)
            _GS._end_turn = _counted  # type: ignore[assignment]
            awbw_to_engine = {AWBW_P0: 0, AWBW_P1: 1}
            action = _capt_action(
                building_row=0, building_col=2,
                buildings_team=str(AWBW_P1),            # defender id
            )
            apply_oracle_action_json(
                st, action, awbw_to_engine,
                envelope_awbw_player_id=AWBW_P0,        # capturer's envelope
            )
        finally:
            _GS._end_turn = orig_end_turn  # type: ignore[assignment]

        self.assertEqual(st.active_player, 0,
                         "P0 must remain active after a contested Capt envelope")
        self.assertEqual(end_turn_calls["n"], 0,
                         "No END_TURN must fire when envelope already aligns "
                         "with the engine's active player")

    def test_sami_scop_capt_opponent_property_preserves_scop_active(self) -> None:
        """Sami P0 with SCOP active captures opponent property — scop_active must
        survive the Capt envelope. Sami SCOP grants +1 movement to infantry
        (analogous risk to Koal COP if seat-flip ever fires)."""
        st = _build_state(
            co_p0_id=8, co_p1_id=1,                    # Sami vs Andy
            width=4,
            capturer_player=0, capturer_pos=(0, 1), capturer_type=UnitType.INFANTRY,
            property_pos=(0, 2), property_terrain_id=BLUE_MOON_CITY,
            property_owner=1,
            active_player=0,
            scop_active_p0=True,
        )
        awbw_to_engine = {AWBW_P0: 0, AWBW_P1: 1}
        action = _capt_action(
            building_row=0, building_col=2,
            buildings_team=str(AWBW_P1),
        )
        apply_oracle_action_json(
            st, action, awbw_to_engine,
            envelope_awbw_player_id=AWBW_P0,
        )
        self.assertEqual(st.active_player, 0)
        self.assertTrue(st.co_states[0].scop_active,
                        "Sami scop_active must survive contested Capt envelope")
        self.assertFalse(st.co_states[1].cop_active)
        self.assertFalse(st.co_states[1].scop_active)


if __name__ == "__main__":
    unittest.main()
