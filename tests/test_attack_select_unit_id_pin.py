"""Phase 11J-F5-OCCUPANCY: ``select_unit_id`` on ATTACK pins the striker on duplicate ``pos``.

``get_unit_at`` returns the first alive unit on a tile; after oracle ``_move_unit_forced``,
two friendly units can share ``unit_pos``. ``get_unit_at_oracle_id`` + ``Action.select_unit_id``
disambiguate (same mechanism as ``SELECT_UNIT``).
"""
from __future__ import annotations

import pytest

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import Unit, UnitType, UNIT_STATS

PLAIN = 1


def _make_state(width: int, height: int) -> GameState:
    md = MapData(
        map_id=0,
        name="f5_occ",
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
    hp: int = 100,
) -> Unit:
    stats = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
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


def test_attack_select_unit_id_resolves_striker_on_duplicate_tile():
    """Oracle duplicate ``pos``: BLACK_BOAT listed before MED_TANK so ``get_unit_at`` would pick the boat."""
    state = _make_state(width=3, height=1)
    _spawn(state, UnitType.BLACK_BOAT, 0, (0, 0), unit_id=101)
    striker = _spawn(state, UnitType.MED_TANK, 0, (0, 0), unit_id=100)
    defender = _spawn(state, UnitType.TANK, 1, (0, 1), unit_id=200, hp=100)
    state.selected_move_pos = (0, 0)

    assert state.get_unit_at(0, 0) is state.units[0][0]

    with pytest.raises(ValueError, match="not in attack range"):
        state._apply_attack(
            Action(
                ActionType.ATTACK,
                unit_pos=(0, 0),
                move_pos=(0, 0),
                target_pos=(0, 1),
                select_unit_id=None,
            )
        )

    def_hp_before = defender.hp
    state._apply_attack(
        Action(
            ActionType.ATTACK,
            unit_pos=(0, 0),
            move_pos=(0, 0),
            target_pos=(0, 1),
            select_unit_id=int(striker.unit_id),
        )
    )
    assert defender.hp < def_hp_before
    assert striker.is_alive


def test_attack_select_unit_id_none_single_occupant_matches_legacy():
    """No stack: ``select_unit_id=None`` still resolves the lone attacker (legacy tests / seam paths)."""
    state = _make_state(width=3, height=1)
    atk = _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=1)
    tgt = _spawn(state, UnitType.INFANTRY, 1, (0, 1), unit_id=2, hp=50)
    state.active_player = 0
    state.selected_move_pos = (0, 0)
    hp0 = tgt.hp
    state._apply_attack(
        Action(
            ActionType.ATTACK,
            unit_pos=(0, 0),
            move_pos=(0, 0),
            target_pos=(0, 1),
            select_unit_id=None,
        )
    )
    assert tgt.hp < hp0
    assert atk.is_alive


def test_attack_select_unit_id_dead_oracle_id_falls_back_to_get_unit_at():
    """``get_unit_at_oracle_id`` skips dead units; pin to a dead ``unit_id`` falls back to ``get_unit_at``."""
    state = _make_state(width=3, height=1)
    _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=100, hp=0)
    alive_same_tile = _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=101)
    tgt = _spawn(state, UnitType.INFANTRY, 1, (0, 1), unit_id=200, hp=50)
    state.active_player = 0
    state.selected_move_pos = (0, 0)
    assert state.get_unit_at(0, 0) is alive_same_tile
    hp0 = tgt.hp
    state._apply_attack(
        Action(
            ActionType.ATTACK,
            unit_pos=(0, 0),
            move_pos=(0, 0),
            target_pos=(0, 1),
            select_unit_id=100,
        )
    )
    assert tgt.hp < hp0
    assert alive_same_tile.is_alive


def test_attack_select_unit_id_wrong_tile_falls_back_to_get_unit_at():
    """Pin id refers to a live unit elsewhere — no match at ``unit_pos``; use tile scan."""
    state = _make_state(width=4, height=1)
    on_tile = _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=10)
    elsewhere = _spawn(state, UnitType.INFANTRY, 0, (0, 3), unit_id=99)
    tgt = _spawn(state, UnitType.INFANTRY, 1, (0, 1), unit_id=20, hp=50)
    state.active_player = 0
    state.selected_move_pos = (0, 0)
    hp0 = tgt.hp
    state._apply_attack(
        Action(
            ActionType.ATTACK,
            unit_pos=(0, 0),
            move_pos=(0, 0),
            target_pos=(0, 1),
            select_unit_id=int(elsewhere.unit_id),
        )
    )
    assert tgt.hp < hp0
    assert on_tile.is_alive
