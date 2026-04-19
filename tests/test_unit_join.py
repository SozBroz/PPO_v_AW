"""AWBW-style join: move same-type ally onto injured ally to merge."""

from __future__ import annotations

import pytest

from engine.action import Action, ActionType, ActionStage, get_legal_actions, units_can_join
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from rl.env import _JOIN_IDX, _action_to_flat, _flat_to_action

from server.play_human import MAPS_DIR, POOL_PATH


def _empty_two_seat_state():
    m = load_map(123858, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 1, tier_name="T3")
    s.units[0] = []
    s.units[1] = []
    st = UNIT_STATS[UnitType.INFANTRY]
    partner = Unit(
        UnitType.INFANTRY,
        0,
        50,
        st.max_ammo,
        st.max_fuel,
        (10, 10),
        False,
        [],
        False,
        20,
        1,
    )
    mover = Unit(
        UnitType.INFANTRY,
        0,
        60,
        st.max_ammo,
        st.max_fuel,
        (10, 9),
        False,
        [],
        False,
        20,
        2,
    )
    s.units[0].extend([partner, mover])
    s.active_player = 0
    s.funds[0] = 0
    s.action_stage = ActionStage.SELECT
    s.selected_unit = None
    s.selected_move_pos = None
    return s, mover, partner


def test_units_can_join_requires_injured_partner():
    st = UNIT_STATS[UnitType.INFANTRY]
    mover = Unit(UnitType.INFANTRY, 0, 100, st.max_ammo, st.max_fuel, (0, 0), False, [], False, 20, 1)
    injured = Unit(UnitType.INFANTRY, 0, 50, st.max_ammo, st.max_fuel, (0, 1), False, [], False, 20, 2)
    assert units_can_join(mover, injured)

    full = Unit(UnitType.INFANTRY, 0, 100, st.max_ammo, st.max_fuel, (0, 1), False, [], False, 20, 3)
    assert not units_can_join(mover, full)


def test_join_merges_hp_and_gold_aw_formula():
    s, mover, partner = _empty_two_seat_state()
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=mover.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=mover.pos, move_pos=partner.pos))
    joins = [a for a in get_legal_actions(s) if a.action_type == ActionType.JOIN]
    assert len(joins) == 1
    s.step(joins[0])

    assert len(s.units[0]) == 1
    u = s.units[0][0]
    assert u.pos == (10, 10)
    assert u.hp == 100  # 50 + 60 capped
    # display 5 + 6 - 10 = 1 bar excess; infantry 1000 -> 100 per bar
    assert s.funds[0] == 100


def test_join_flat_roundtrip():
    s, mover, partner = _empty_two_seat_state()
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=mover.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=mover.pos, move_pos=partner.pos))
    legal = get_legal_actions(s)
    j = next(a for a in legal if a.action_type == ActionType.JOIN)
    assert _action_to_flat(j) == _JOIN_IDX
    assert _flat_to_action(_JOIN_IDX, s).action_type == ActionType.JOIN


def test_illegal_wait_on_join_tile():
    s, mover, partner = _empty_two_seat_state()
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=mover.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=mover.pos, move_pos=partner.pos))
    with pytest.raises(ValueError, match="JOIN"):
        s.step(Action(ActionType.WAIT, unit_pos=mover.pos, move_pos=partner.pos))
