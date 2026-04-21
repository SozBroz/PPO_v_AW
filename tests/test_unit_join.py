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


def test_units_can_join_requires_at_least_one_damaged():
    """AWBW: JOIN allowed when *either* the mover or partner is below full HP.
    Only refused when both are at 100. (Previously the engine wrongly required
    an injured partner; that was restrictive vs AWBW.)"""
    st = UNIT_STATS[UnitType.INFANTRY]
    full_a   = Unit(UnitType.INFANTRY, 0, 100, st.max_ammo, st.max_fuel, (0, 0), False, [], False, 20, 1)
    full_b   = Unit(UnitType.INFANTRY, 0, 100, st.max_ammo, st.max_fuel, (0, 1), False, [], False, 20, 2)
    injured  = Unit(UnitType.INFANTRY, 0,  50, st.max_ammo, st.max_fuel, (0, 1), False, [], False, 20, 3)
    damaged_mover = Unit(UnitType.INFANTRY, 0, 60, st.max_ammo, st.max_fuel, (0, 0), False, [], False, 20, 4)

    assert units_can_join(full_a, injured)         # full mover + injured partner
    assert units_can_join(damaged_mover, full_b)   # damaged mover + full partner
    assert units_can_join(damaged_mover, injured)  # both damaged
    assert not units_can_join(full_a, full_b)      # both full -> refused


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
    # STEP-GATE (Phase 3) rejects this WAIT before ``_apply_wait`` reaches
    # its JOIN-specific ValueError; either error path is an acceptable
    # rejection of the illegal move.
    with pytest.raises(ValueError, match="JOIN|get_legal_actions"):
        s.step(Action(ActionType.WAIT, unit_pos=mover.pos, move_pos=partner.pos))
