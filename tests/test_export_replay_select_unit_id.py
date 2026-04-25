"""Replay export must restore ``select_unit_id`` from ``full_trace`` and pin
``moving_unit`` for Move/Fire JSON — otherwise stacked tiles emit the wrong
``units_id`` and the AWBW Replay Player throws ``ReplayMissingUnitException`` on
``MoveUnitAction.SetupAndUpdate`` (logged as ``Failed to setup turn N``).
"""
from __future__ import annotations

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import Unit, UnitType, UNIT_STATS

from tools.export_awbw_replay import P0_PLAYER_ID, P1_PLAYER_ID
from tools.export_awbw_replay_actions import _emit_move_or_fire, _trace_to_action

PLAIN = 1


def _make_state(width: int, height: int) -> GameState:
    md = MapData(
        map_id=0,
        name="export_stack",
        map_type="std",
        terrain=[[PLAIN] * width for _ in range(height)],
        height=height,
        width=width,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(0), make_co_state_safe(0)],
        properties=[],
        turn=1,
        active_player=0,
        action_stage=ActionStage.ACTION,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


def _spawn(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    unit_id: int,
) -> Unit:
    stats = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=100,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


def test_trace_to_action_restores_select_unit_id() -> None:
    entry = {
        "type": "WAIT",
        "player": 0,
        "turn": 1,
        "stage": "ACTION",
        "unit_pos": [0, 0],
        "move_pos": [0, 0],
        "target_pos": None,
        "unit_type": None,
        "select_unit_id": 100,
    }
    a = _trace_to_action(entry)
    assert a.select_unit_id == 100


def test_trace_to_action_select_unit_id_omitted_is_none() -> None:
    entry = {
        "type": "WAIT",
        "player": 0,
        "turn": 1,
        "stage": "ACTION",
        "unit_pos": [0, 0],
        "move_pos": [0, 0],
        "target_pos": None,
        "unit_type": None,
        "select_unit_id": None,
    }
    a = _trace_to_action(entry)
    assert a.select_unit_id is None


def test_emit_wait_move_json_uses_oracle_pinned_unit_on_stack() -> None:
    """``get_unit_at`` would return the boat (inserted first); trace pins the tank."""
    state = _make_state(width=3, height=1)
    _spawn(state, UnitType.BLACK_BOAT, 0, (0, 0), unit_id=101)
    striker = _spawn(state, UnitType.MED_TANK, 0, (0, 0), unit_id=100)
    state.selected_move_pos = (0, 0)

    assert state.get_unit_at(0, 0).unit_id == 101

    pid_of = {0: P0_PLAYER_ID, 1: P1_PLAYER_ID}
    action = Action(
        ActionType.WAIT,
        unit_pos=(0, 0),
        move_pos=(0, 0),
        select_unit_id=int(striker.unit_id),
    )
    payload = _emit_move_or_fire(state, action, pid_of, P0_PLAYER_ID, P1_PLAYER_ID)
    assert payload is not None
    assert payload["action"] == "Move"
    assert payload["unit"]["global"]["units_id"] == int(striker.unit_id)


def test_emit_missiles_attack_fire_envelope_uses_missiles_name_and_defender_lookup() -> None:
    """Missiles (AA indirect) Fire JSON must use AWBW ``Missiles`` name and a valid
    defender block — regressions here crash the desktop replay viewer mid-game.
    """
    state = _make_state(width=15, height=15)
    aa = _spawn(state, UnitType.MISSILES, 0, (5, 5), unit_id=500)
    _spawn(state, UnitType.B_COPTER, 1, (5, 10), unit_id=501)
    state.active_player = 0
    state.action_stage = ActionStage.ACTION
    state.selected_unit = aa
    state.selected_move_pos = (5, 5)

    pid_of = {0: P0_PLAYER_ID, 1: P1_PLAYER_ID}
    action = Action(
        ActionType.ATTACK,
        unit_pos=(5, 5),
        move_pos=(5, 5),
        target_pos=(5, 10),
        select_unit_id=500,
    )
    payload = _emit_move_or_fire(state, action, pid_of, P0_PLAYER_ID, P1_PLAYER_ID)
    assert payload is not None
    assert payload["action"] == "Fire"
    g = payload["Move"]["unit"]["global"]
    assert g["units_name"] == "Missiles"
    assert g["units_symbol"] == "Missiles"
    def_hp = payload["Fire"]["combatInfoVision"]["global"]["combatInfo"]["defender"][
        "units_hit_points"
    ]
    assert def_hp in (0, 1)
