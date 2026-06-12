"""Comm tower / lab capturability + combat ghost purge + stale-terminal penalty.

Pins the fixes for:
1. Comm towers and labs are capturable (AWBW-correct) and the WAIT mask
   applies on them like any other capturable property.
2. Combat dead are pruned from rosters immediately (no hp==0 ghosts in the
   RL path), and army-wipe evaluation counts alive units only.
3. The NN encoder never paints hp==0 ghosts into unit-presence channels.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engine.action import Action, ActionStage, ActionType, get_legal_actions
from engine.commander_wars_capture import (
    is_capturable_property_at,
    is_foot_capture_mandatory_tile,
)
from engine.game import PropertyState
from engine.unit import UnitType
from test_engine_awbw_subset import _blank_state as _fresh_state, _spawn


def _add_prop(s, r, c, *, owner=None, is_comm_tower=False, is_lab=False, tid=133):
    s.map_data.terrain[r][c] = tid
    s.properties.append(
        PropertyState(
            terrain_id=tid,
            row=r,
            col=c,
            owner=owner,
            capture_points=20,
            is_hq=False,
            is_lab=is_lab,
            is_comm_tower=is_comm_tower,
            is_base=False,
            is_airport=False,
            is_port=False,
        )
    )
    return s.properties[-1]


# ---------------------------------------------------------------------------
# 1. Tower / lab capturability
# ---------------------------------------------------------------------------

def test_neutral_comm_tower_is_capturable_and_wait_masked():
    s = _fresh_state()
    r, c = 2, 2
    _add_prop(s, r, c, owner=None, is_comm_tower=True, tid=133)

    assert is_capturable_property_at(s, 0, (r, c)) is True
    assert is_foot_capture_mandatory_tile(s, 0, (r, c)) is True

    inf = _spawn(s, UnitType.INFANTRY, 0, (r, c))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos, move_pos=inf.pos))
    legal = get_legal_actions(s)
    types = {a.action_type for a in legal}
    assert ActionType.CAPTURE in types, "CAPTURE must be offered on a comm tower"
    assert ActionType.WAIT not in types, "WAIT must be masked on a capturable tower"


def test_enemy_lab_is_capturable():
    s = _fresh_state()
    r, c = 2, 3
    _add_prop(s, r, c, owner=1, is_lab=True, tid=145)

    assert is_capturable_property_at(s, 0, (r, c)) is True
    assert is_foot_capture_mandatory_tile(s, 0, (r, c)) is True


def test_own_comm_tower_not_capture_mandatory():
    s = _fresh_state()
    r, c = 2, 4
    _add_prop(s, r, c, owner=0, is_comm_tower=True, tid=134)

    assert is_capturable_property_at(s, 0, (r, c)) is False
    assert is_foot_capture_mandatory_tile(s, 0, (r, c)) is False


def test_capture_flips_comm_tower_ownership():
    s = _fresh_state()
    r, c = 2, 5
    prop = _add_prop(s, r, c, owner=None, is_comm_tower=True, tid=133)
    s.map_data.country_to_player = {1: 0, 2: 1}

    inf = _spawn(s, UnitType.INFANTRY, 0, (r, c))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos, move_pos=inf.pos))
    s.step(Action(ActionType.CAPTURE, unit_pos=inf.pos, move_pos=inf.pos))
    # 10 HP infantry: 10 of 20 points on first attempt — not flipped yet.
    assert prop.owner is None
    assert prop.capture_points == 10


# ---------------------------------------------------------------------------
# 2. Combat ghost purge
# ---------------------------------------------------------------------------

def test_combat_kill_prunes_dead_from_roster():
    s = _fresh_state()
    att = _spawn(s, UnitType.MED_TANK, 0, (5, 5))
    vic = _spawn(s, UnitType.INFANTRY, 1, (5, 6))
    vic.hp = 1  # guaranteed kill

    s.step(Action(ActionType.SELECT_UNIT, unit_pos=att.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=att.pos, move_pos=att.pos))
    s.step(Action(
        ActionType.ATTACK, unit_pos=att.pos, move_pos=att.pos, target_pos=vic.pos,
    ))

    assert vic.hp == 0
    assert all(u.is_alive for u in s.units[1]), "hp==0 ghost left on roster"


def test_army_wipe_ignores_ghosts():
    s = _fresh_state()
    # Seat 1's only roster entry is a ghost; seat 0 has a live unit.
    ghost = _spawn(s, UnitType.INFANTRY, 1, (8, 8))
    ghost.hp = 0
    _spawn(s, UnitType.INFANTRY, 0, (1, 1))

    s._evaluate_army_wipe_after_combat()
    assert s.done is True
    assert s.winner == 0
    assert s.win_reason == "army_wipe"


# ---------------------------------------------------------------------------
# 3. Stale terminal intents are penalized, not silently swallowed
# ---------------------------------------------------------------------------

def test_stale_terminal_intent_recovers_without_illegal_charge():
    """Stale terminals recover cleanly and are NOT charged as illegal genes.

    Charging them (0.02 each, a full turn's phi gain) purged all combat from
    the gene pool: seed-13 A/B showed a 31-day battle collapse to a 9-day
    zero-Fire capture race. Recovery alone fixes the ghost/sitting bug.
    """
    from rl.rhea import RheaConfig, RheaPlanner, UnitIntent

    class _DummyFitness:
        pass

    s = _fresh_state()
    inf = _spawn(s, UnitType.INFANTRY, 0, (4, 4))
    # Intent encodes a CAPTURE on a plain tile (e.g. the property flipped or
    # the sim diverged) — no CAPTURE is legal there and no substitute exists.
    intent = UnitIntent(
        unit_pos=(4, 4),
        move_dest=(4, 4),
        action_type=ActionType.CAPTURE,
        target_pos=None,
    )
    planner = RheaPlanner(_DummyFitness(), RheaConfig(seed=1))
    actions: list = []
    ok, illegal = planner._execute_unit_intent(s, intent, actions)
    assert ok is True
    assert illegal == 0, "recovered stale terminal must not be charged"
    # Recovery: the unit must have finished its turn with a real action.
    assert inf.moved is True
    assert actions[-1].action_type == ActionType.WAIT


# ---------------------------------------------------------------------------
# 4. Encoder ghost guard
# ---------------------------------------------------------------------------

def test_encoder_skips_dead_units():
    from rl.encoder import encode_state

    s = _fresh_state()
    live = _spawn(s, UnitType.INFANTRY, 0, (4, 4))
    ghost = _spawn(s, UnitType.INFANTRY, 1, (6, 6))
    ghost.hp = 0

    spatial, _scalars = encode_state(s, observer=0)
    # Enemy block channels are 14..27; ghost tile must stay dark.
    assert float(spatial[6, 6, :28].sum()) == 0.0, "ghost painted into presence channels"
    assert float(spatial[4, 4, :14].sum()) > 0.0, "live unit missing from own block"
