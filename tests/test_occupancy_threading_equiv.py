"""Phase 2d: shared per-call occupancy across get_legal_actions and helpers."""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.action as action_module
from engine.action import (
    Action,
    ActionStage,
    ActionType,
    _get_action_actions,
    _get_move_actions,
    _get_select_actions,
    _build_occupancy,
    compute_reachable_costs,
    get_attack_targets,
    get_legal_actions,
)
from engine.game import GameState
from engine.unit import Unit, UnitType
from tests.test_compute_reachable_costs_occupancy_equiv import (
    _make_unit,
    _rect_map,
    _scenario_states,
)
from engine.game import make_initial_state
from engine.map_loader import load_map
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
MAP_166877 = 166877

_CORPUS_DIR = ROOT / "tests" / "data" / "legal_actions_equivalence_corpus"
_PLAIN = 1


def _corpus_states() -> list[GameState]:
    pkls = sorted(_CORPUS_DIR.glob("*.pkl"))
    out: list[GameState] = []
    for p in pkls:
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


def _get_legal_actions_unthreaded(state: GameState) -> list[Action]:
    """Same as get_legal_actions but without top-level _build_occupancy (helpers use default)."""
    player = state.active_player
    if state.action_stage == ActionStage.SELECT:
        actions = _get_select_actions(state, player)
    elif state.action_stage == ActionStage.MOVE:
        actions = _get_move_actions(state, player)
    elif state.action_stage == ActionStage.ACTION:
        actions = _get_action_actions(state, player)
    else:
        actions = []
    for _a in actions:
        if _a.action_type not in action_module._RL_LEGAL_ACTION_TYPES:
            raise AssertionError(
                f"get_legal_actions emitted non-RL-legal action {_a.action_type.name}"
            )
    return actions


def _synthetic_select_move_action_states() -> list[GameState]:
    """Small map with several units; SELECT, one MOVE, one ACTION snapshot."""
    out: list[GameState] = []
    p5 = _rect_map(991_050, "occ2d_smoke", 5, 5, _PLAIN)
    st = make_initial_state(p5, 1, 1, starting_funds=50000, tier_name="T2", replay_first_mover=0)
    st.units = {0: [], 1: []}
    uid = 1
    u0 = _make_unit(UnitType.INFANTRY, 0, (2, 2), uid)
    uid += 1
    u1 = _make_unit(UnitType.TANK, 1, (2, 4), uid)
    uid += 1
    u2 = _make_unit(UnitType.INFANTRY, 0, (0, 0), uid)
    st.units[0].extend([u0, u2])
    st.units[1].append(u1)
    st.action_stage = ActionStage.SELECT
    st.selected_unit = None
    st.selected_move_pos = None
    out.append(copy.deepcopy(st))

    st_m = copy.deepcopy(st)
    st_m.active_player = 0
    st_m.step(Action(ActionType.SELECT_UNIT, unit_pos=u0.pos))
    out.append(copy.deepcopy(st_m))

    if st_m.action_stage == ActionStage.MOVE and st_m.selected_unit is not None:
        reach = compute_reachable_costs(st_m, st_m.selected_unit)
        alt = [p for p in reach if p != st_m.selected_unit.pos]
        if alt:
            mp = alt[0]
            st_a = copy.deepcopy(st_m)
            st_a.step(
                Action(
                    ActionType.SELECT_UNIT,
                    unit_pos=st_a.selected_unit.pos,
                    move_pos=mp,
                )
            )
            if st_a.action_stage == ActionStage.ACTION:
                out.append(copy.deepcopy(st_a))
    return out


def test_get_legal_actions_byte_identical_with_threaded_occupancy() -> None:
    states: list[GameState] = []
    states.extend(_synthetic_select_move_action_states())
    for s, _u, _tag in _scenario_states():
        states.append(copy.deepcopy(s))
    for s in _corpus_states()[:8]:
        states.append(copy.deepcopy(s))
    assert states, "need at least one state"

    for st in states:
        s1 = copy.deepcopy(st)
        s2 = copy.deepcopy(st)
        got = get_legal_actions(s1)
        ref = _get_legal_actions_unthreaded(s2)
        assert got == ref, f"stage={st.action_stage!r} actions differ"


def test_get_attack_targets_with_explicit_occupancy_matches_default() -> None:
    try:
        md = load_map(MAP_166877, MAP_POOL, MAPS_DIR)
    except Exception:
        pytest.skip("map 166877 unavailable")
    st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st.units = {0: [], 1: []}
    t = _make_unit(UnitType.TANK, 0, (2, 2), 1)
    e = _make_unit(UnitType.INFANTRY, 1, (2, 4), 2)
    st.units[0].append(t)
    st.units[1].append(e)
    for mp in (t.pos, (2, 3), (3, 2)):
        occ = _build_occupancy(st)
        a = get_attack_targets(st, t, mp)
        b = get_attack_targets(st, t, mp, occupancy=occ)
        assert a == b, f"move_pos={mp}"


def test_compute_reachable_costs_with_explicit_occupancy_matches_2c() -> None:
    for state, unit, _label in _scenario_states():
        occ = _build_occupancy(state)
        d0 = compute_reachable_costs(state, unit)
        d1 = compute_reachable_costs(state, unit, occupancy=occ)
        assert d0 == d1, f"{_label} unit {unit.unit_id}"
    for st in _corpus_states()[:3]:
        for p in (0, 1):
            for u in list(st.units.get(p, ())):
                if not u.is_alive:
                    continue
                occ = _build_occupancy(st)
                assert compute_reachable_costs(st, u) == compute_reachable_costs(
                    st, u, occupancy=occ
                )


def test_external_caller_signatures_unchanged() -> None:
    try:
        md = load_map(MAP_166877, MAP_POOL, MAPS_DIR)
    except Exception:
        pytest.skip("map 166877 unavailable")
    st = make_initial_state(md, 1, 7, starting_funds=50000, tier_name="T2", replay_first_mover=0)
    st.action_stage = ActionStage.SELECT
    st.active_player = 0
    _get_select_actions(st, 0)
    u = st.units[0][0] if st.units[0] else st.units[1][0]
    st.active_player = u.player
    st.step(Action(ActionType.SELECT_UNIT, unit_pos=u.pos))
    _get_move_actions(st, st.active_player)
    s = st.selected_unit
    if s is not None and st.action_stage == ActionStage.MOVE:
        r = compute_reachable_costs(st, s)
        mp = next(iter(set(r.keys()) - {s.pos}), s.pos)
        st.step(Action(ActionType.SELECT_UNIT, unit_pos=s.pos, move_pos=mp))
    if st.action_stage == ActionStage.ACTION and st.selected_unit and st.selected_move_pos:
        _get_action_actions(st, st.active_player)

    any_u = st.units[0][0] if st.units[0] else st.units[1][0]
    compute_reachable_costs(st, any_u)
    if st.selected_unit is not None and st.selected_move_pos is not None:
        get_attack_targets(st, st.selected_unit, st.selected_move_pos)


def test_occupancy_built_once_per_get_legal_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    n = {"c": 0}
    real = action_module._build_occupancy

    def _counting(state: GameState) -> dict[tuple[int, int], Unit]:
        n["c"] += 1
        return real(state)

    try:
        md = load_map(MAP_166877, MAP_POOL, MAPS_DIR)
    except Exception:
        pytest.skip("map 166877 unavailable")
    st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2", replay_first_mover=0)
    u = st.units[0][0] if st.units[0] else st.units[1][0]
    st.active_player = u.player
    st.action_stage = ActionStage.SELECT
    monkeypatch.setattr(action_module, "_build_occupancy", _counting)
    get_legal_actions(st)
    assert n["c"] == 1, f"expected single _build_occupancy, got {n['c']}"

    n2 = {"c": 0}

    def _counting2(state: GameState) -> dict[tuple[int, int], Unit]:
        n2["c"] += 1
        return real(state)

    st2 = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2", replay_first_mover=0)
    u2 = st2.units[0][0] if st2.units[0] else st2.units[1][0]
    st2.active_player = u2.player
    st2.step(Action(ActionType.SELECT_UNIT, unit_pos=u2.pos))
    monkeypatch.setattr(action_module, "_build_occupancy", _counting2)
    get_legal_actions(st2)
    assert n2["c"] == 1
