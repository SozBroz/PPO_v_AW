"""MOVE-stage flat indices: one slot per destination (see move_encoding_redesign.md)."""

from __future__ import annotations

import pytest

from engine.action import ActionStage, ActionType, get_legal_actions
from rl.env import (
    ACTION_SPACE_SIZE,
    _MOVE_OFFSET,
    _action_to_flat,
    _flat_to_action,
    _get_action_mask,
)
from tests.test_encoder_equivalence import _s1


def _reach_move_stage():
    """Return (state, legal_moves) after one SELECT_UNIT on encoder-equivalence s1."""
    state = _s1()
    legal = get_legal_actions(state)
    pick = next(
        (
            a
            for a in legal
            if a.action_type == ActionType.SELECT_UNIT and a.move_pos is None
        ),
        None,
    )
    if pick is None:
        return None
    state, _r, done = state.step(pick)
    if done:
        return None
    legal = get_legal_actions(state)
    if state.action_stage != ActionStage.MOVE:
        return None
    return state, legal


def test_move_offset_constant():
    assert _MOVE_OFFSET == 1818


def test_move_stage_encodes_destination_not_source():
    hit = _reach_move_stage()
    assert hit is not None
    state, legal = hit
    move_actions = [
        a
        for a in legal
        if a.action_type == ActionType.SELECT_UNIT and a.move_pos is not None
    ]
    assert len(move_actions) >= 1
    flats = {_action_to_flat(a, state) for a in move_actions}
    assert len(flats) == len(move_actions)
    for a in move_actions:
        r, c = a.move_pos
        assert _action_to_flat(a, state) == _MOVE_OFFSET + r * 30 + c


def test_mask_move_slice_counts_match_legal_moves():
    hit = _reach_move_stage()
    assert hit is not None
    state, legal = hit
    move_actions = [
        a
        for a in legal
        if a.action_type == ActionType.SELECT_UNIT and a.move_pos is not None
    ]
    mask = _get_action_mask(state)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    n_bits = int(mask[_MOVE_OFFSET : _MOVE_OFFSET + 900].sum())
    assert n_bits == len(move_actions)


def test_flat_to_action_roundtrip_move():
    hit = _reach_move_stage()
    assert hit is not None
    state, legal = hit
    for a in legal:
        if a.action_type != ActionType.SELECT_UNIT or a.move_pos is None:
            continue
        idx = _action_to_flat(a, state)
        back = _flat_to_action(idx, state, legal=legal)
        assert back is not None
        assert back.move_pos == a.move_pos
        assert back.unit_pos == a.unit_pos


def test_select_stage_still_uses_unit_tile():
    state = _s1()
    assert state.action_stage == ActionStage.SELECT
    legal = get_legal_actions(state)
    picks = [
        a
        for a in legal
        if a.action_type == ActionType.SELECT_UNIT and a.move_pos is None
    ]
    assert picks
    a = picks[0]
    r, c = a.unit_pos
    assert _action_to_flat(a, state) == 3 + r * 30 + c
