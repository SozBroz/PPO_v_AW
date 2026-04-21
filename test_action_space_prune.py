"""Action-space pruning: END_TURN gating and infantry/mech WAIT on properties.

Two rules enforced in ``engine/action.py``:

  1. ``END_TURN`` at SELECT stage is only legal once every friendly unit has
     ``moved=True`` (or the player has no units at all). The no-op path is
     still available as ``SELECT_UNIT -> WAIT`` on the unit's own tile.
  2. A capture-capable unit (infantry/mech) on a neutral or enemy capturable
     property cannot ``WAIT`` when ``CAPTURE`` is available (same rule for
     mid-capture). ``ATTACK`` alone does not remove ``WAIT``. To leave the
     tile without capturing, use ``SELECT`` / ``MOVE`` on a later turn.
     Missile silos (non-property) and owned buildings are unaffected; if
     ``CAPTURE`` is not offered the unit may still ``WAIT`` so it is never
     deadlocked.
"""
from __future__ import annotations

import unittest

from engine.action import (
    Action, ActionType, ActionStage,
    get_legal_actions,
)
from engine.game import GameState, IllegalActionError, make_initial_state
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS


# Terrain IDs (see engine/terrain.py)
PLAIN          = 1
NEUTRAL_CITY   = 34
NEUTRAL_BASE   = 35
MISSILE_SILO   = 111


def _build_map(terrain: list[list[int]], properties: list[PropertyState]) -> MapData:
    return MapData(
        map_id=999_998,
        name="prune_action_space_test",
        map_type="std",
        terrain=terrain,
        height=len(terrain),
        width=len(terrain[0]),
        cap_limit=999,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=properties,
        hq_positions={},
        lab_positions={},
        country_to_player={},
        predeployed_specs=[],
    )


def _neutral_city(row: int, col: int, *, capture_points: int = 20) -> PropertyState:
    return PropertyState(
        terrain_id=NEUTRAL_CITY,
        row=row, col=col,
        owner=None,
        capture_points=capture_points,
        is_hq=False, is_lab=False, is_comm_tower=False,
        is_base=False, is_airport=False, is_port=False,
    )


def _owned_city(row: int, col: int, owner: int) -> PropertyState:
    return PropertyState(
        terrain_id=NEUTRAL_CITY,
        row=row, col=col,
        owner=owner,
        capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False,
        is_base=False, is_airport=False, is_port=False,
    )


def _make_unit(
    state: GameState,
    unit_type: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    hp: int = 100,
) -> Unit:
    stats = UNIT_STATS[unit_type]
    u = Unit(
        unit_type=unit_type,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=state._allocate_unit_id(),
    )
    state.units[player].append(u)
    return u


def _fresh_state(terrain: list[list[int]], properties: list[PropertyState]) -> GameState:
    md = _build_map(terrain, properties)
    st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
    st.units = {0: [], 1: []}
    st.active_player = 0
    return st


def _select_and_move(state: GameState, unit: Unit, dest: tuple[int, int]) -> None:
    state.action_stage      = ActionStage.SELECT
    state.selected_unit     = None
    state.selected_move_pos = None
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos))
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=dest))


# ---------------------------------------------------------------------------
# 1. END_TURN gating
# ---------------------------------------------------------------------------

class TestEndTurnGating(unittest.TestCase):
    """END_TURN hidden while any friendly unit still has moved=False."""

    def _state_with_unit(self) -> tuple[GameState, Unit]:
        terrain = [[PLAIN] * 5 for _ in range(5)]
        st = _fresh_state(terrain, [])
        unit = _make_unit(st, UnitType.INFANTRY, 0, (2, 2))
        return st, unit

    def test_end_turn_absent_when_unit_unmoved(self) -> None:
        st, _ = self._state_with_unit()
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertNotIn(ActionType.END_TURN, types)
        self.assertIn(ActionType.SELECT_UNIT, types)

    def test_end_turn_present_after_unit_moves(self) -> None:
        st, unit = self._state_with_unit()
        unit.moved = True
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.END_TURN, types)

    def test_end_turn_present_when_no_units(self) -> None:
        terrain = [[PLAIN] * 3 for _ in range(3)]
        st = _fresh_state(terrain, [])
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.END_TURN, types)

    def test_wait_in_place_unlocks_end_turn(self) -> None:
        """The no-op path (SELECT_UNIT -> WAIT on own tile) still works."""
        st, unit = self._state_with_unit()
        _select_and_move(st, unit, unit.pos)
        st.step(Action(ActionType.WAIT, unit_pos=unit.pos, move_pos=unit.pos))
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.END_TURN, types)


# ---------------------------------------------------------------------------
# 2. Infantry/mech WAIT pruning on contested properties
# ---------------------------------------------------------------------------

class TestWaitPruningOnProperty(unittest.TestCase):
    """Capture-capable units must ATTACK or CAPTURE on neutral/enemy tiles."""

    def test_infantry_on_neutral_city_loses_wait(self) -> None:
        terrain = [[PLAIN] * 5 for _ in range(5)]
        terrain[2][2] = NEUTRAL_CITY
        st = _fresh_state(terrain, [_neutral_city(2, 2)])
        inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))

        _select_and_move(st, inf, (2, 2))
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.CAPTURE, types)
        self.assertNotIn(ActionType.WAIT, types)

    def test_infantry_on_partially_capped_neutral_city_loses_wait(self) -> None:
        """Mid-capture: still no WAIT while CAPTURE is legal."""
        terrain = [[PLAIN] * 5 for _ in range(5)]
        terrain[2][2] = NEUTRAL_CITY
        st = _fresh_state(terrain, [_neutral_city(2, 2, capture_points=12)])
        inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))

        _select_and_move(st, inf, (2, 2))
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.CAPTURE, types)
        self.assertNotIn(ActionType.WAIT, types)

    def test_mech_on_enemy_base_loses_wait(self) -> None:
        terrain = [[PLAIN] * 5 for _ in range(5)]
        terrain[2][2] = NEUTRAL_BASE
        enemy_base = PropertyState(
            terrain_id=NEUTRAL_BASE, row=2, col=2, owner=1, capture_points=20,
            is_hq=False, is_lab=False, is_comm_tower=False,
            is_base=True, is_airport=False, is_port=False,
        )
        st = _fresh_state(terrain, [enemy_base])
        mech = _make_unit(st, UnitType.MECH, 0, (2, 3))

        _select_and_move(st, mech, (2, 2))
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.CAPTURE, types)
        self.assertNotIn(ActionType.WAIT, types)

    def test_infantry_on_owned_city_keeps_wait(self) -> None:
        terrain = [[PLAIN] * 5 for _ in range(5)]
        terrain[2][2] = NEUTRAL_CITY
        st = _fresh_state(terrain, [_owned_city(2, 2, owner=0)])
        inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))

        _select_and_move(st, inf, (2, 2))
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.WAIT, types)
        self.assertNotIn(ActionType.CAPTURE, types)

    def test_infantry_on_silo_keeps_wait(self) -> None:
        """Missile silos are not properties — capture is impossible, so WAIT stays."""
        terrain = [[PLAIN] * 5 for _ in range(5)]
        terrain[2][2] = MISSILE_SILO
        st = _fresh_state(terrain, [])
        inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))

        _select_and_move(st, inf, (2, 2))
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.WAIT, types)
        self.assertNotIn(ActionType.CAPTURE, types)

    def test_tank_on_neutral_city_keeps_wait(self) -> None:
        """Non-capturing units never enter the prune branch."""
        terrain = [[PLAIN] * 5 for _ in range(5)]
        terrain[2][2] = NEUTRAL_CITY
        st = _fresh_state(terrain, [_neutral_city(2, 2)])
        tank = _make_unit(st, UnitType.TANK, 0, (2, 3))

        _select_and_move(st, tank, (2, 2))
        types = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.WAIT, types)
        self.assertNotIn(ActionType.CAPTURE, types)

    def test_step_accepts_hand_crafted_wait_on_neutral_city(self) -> None:
        """Phase 10M: WAIT on capturable tile is pruned from the mask; STEP-GATE
        rejects the same action (canonical engine contract — not AWBW superset
        via ``step`` without ``oracle_mode``)."""
        terrain = [[PLAIN] * 5 for _ in range(5)]
        terrain[2][2] = NEUTRAL_CITY
        prop = _neutral_city(2, 2)
        st = _fresh_state(terrain, [prop])
        inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))

        _select_and_move(st, inf, (2, 2))
        cap_before = prop.capture_points
        with self.assertRaises(IllegalActionError):
            st.step(Action(ActionType.WAIT, unit_pos=inf.pos, move_pos=(2, 2)))
        # Move is staged (unit still on start tile) until a legal ACTION completes.
        self.assertEqual(st.selected_move_pos, (2, 2))
        self.assertIs(st.selected_unit, inf)
        self.assertEqual(prop.capture_points, cap_before)

    def test_step_accepts_wait_on_partially_capped_city(self) -> None:
        """Phase 10M: same STEP-GATE contract as ``test_step_accepts_hand_crafted_wait_on_neutral_city``."""
        terrain = [[PLAIN] * 5 for _ in range(5)]
        terrain[2][2] = NEUTRAL_CITY
        prop = _neutral_city(2, 2, capture_points=8)
        st = _fresh_state(terrain, [prop])
        inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))

        _select_and_move(st, inf, (2, 2))
        cap_before = prop.capture_points
        with self.assertRaises(IllegalActionError):
            st.step(Action(ActionType.WAIT, unit_pos=inf.pos, move_pos=(2, 2)))
        self.assertEqual(st.selected_move_pos, (2, 2))
        self.assertIs(st.selected_unit, inf)
        self.assertEqual(prop.capture_points, cap_before)


if __name__ == "__main__":
    unittest.main()

# Made with Bob
