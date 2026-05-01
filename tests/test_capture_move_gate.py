"""Tests for AWBW_CAPTURE_MOVE_GATE in engine.action._get_move_actions."""

from __future__ import annotations

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    _get_move_actions,
    get_reachable_tiles,
    get_terrain,
    parse_capture_move_gate_env_value,
)
from engine.unit import UNIT_STATS, UnitType
from tests.test_engine_awbw_subset import _blank_state, _spawn


def _capturable_positions_in_reach(state, player: int, reachable: set[tuple[int, int]]):
    out: set[tuple[int, int]] = set()
    for pos in reachable:
        tid = state.map_data.terrain[pos[0]][pos[1]]
        if not get_terrain(tid).is_property:
            continue
        prop = state.get_property_at(*pos)
        if prop is None or prop.is_comm_tower or prop.is_lab:
            continue
        if prop.owner is not None and prop.owner == player:
            continue
        out.add(pos)
    return out


def _find_infantry_adjacent_to_enemy_property(s):
    """Return ((row, col), enemy_prop) for empty ground orthogonally adjacent to P1 property."""
    from engine.terrain import get_move_cost, MOVE_TREAD, INF_PASSABLE

    for target_prop in s.properties:
        if target_prop.owner != 1:
            continue
        if target_prop.is_comm_tower or target_prop.is_lab:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ar, ac = target_prop.row + dr, target_prop.col + dc
            if not (0 <= ar < s.map_data.height and 0 <= ac < s.map_data.width):
                continue
            tid = s.map_data.terrain[ar][ac]
            if get_move_cost(tid, MOVE_TREAD) >= INF_PASSABLE:
                continue
            if s.get_unit_at(ar, ac) is not None:
                continue
            return (ar, ac), target_prop
    return None, None


def _find_infantry_tile_no_capturable_in_reach(s):
    """Empty tile where infantry has several reachable tiles but none are capturable."""
    from engine.terrain import get_move_cost, MOVE_TREAD, INF_PASSABLE

    for r in range(s.map_data.height):
        for c in range(s.map_data.width):
            if s.get_unit_at(r, c) is not None:
                continue
            tid = s.map_data.terrain[r][c]
            if get_move_cost(tid, MOVE_TREAD) >= INF_PASSABLE:
                continue
            inf = _spawn(s, UnitType.INFANTRY, 0, (r, c))
            reach = get_reachable_tiles(s, inf)
            cap = _capturable_positions_in_reach(s, 0, reach)
            if len(reach) > 3 and not cap:
                return (r, c)
            s.units[0] = []
    return None


def _move_positions(actions: list) -> set[tuple[int, int]]:
    return {a.move_pos for a in actions if a.move_pos is not None}


def test_parse_capture_move_gate_env_value() -> None:
    assert parse_capture_move_gate_env_value(None) == 0.0
    assert parse_capture_move_gate_env_value("") == 0.0
    assert parse_capture_move_gate_env_value("false") == 0.0
    assert parse_capture_move_gate_env_value("1") == 1.0
    assert parse_capture_move_gate_env_value("true") == 1.0
    assert parse_capture_move_gate_env_value("0.75") == 0.75
    assert parse_capture_move_gate_env_value("75") == 0.75


def test_capture_move_gate_unset_matches_full_reachable(monkeypatch):
    monkeypatch.delenv("AWBW_CAPTURE_MOVE_GATE", raising=False)

    s = _blank_state()
    pos_prop = _find_infantry_adjacent_to_enemy_property(s)
    assert pos_prop[0] is not None, "fixture map needs P1 property with adjacent empty ground"
    inf_pos, _ = pos_prop
    inf = _spawn(s, UnitType.INFANTRY, 0, inf_pos)
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))
    assert s.action_stage == ActionStage.MOVE

    full = get_reachable_tiles(s, s.selected_unit)
    moves = _get_move_actions(s, 0)
    assert _move_positions(moves) == full


def test_capture_move_gate_on_restricts_to_capturable_properties(monkeypatch):
    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "1")

    s = _blank_state()
    pos_prop = _find_infantry_adjacent_to_enemy_property(s)
    assert pos_prop[0] is not None
    inf_pos, _ = pos_prop
    inf = _spawn(s, UnitType.INFANTRY, 0, inf_pos)
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))

    full = get_reachable_tiles(s, s.selected_unit)
    expected = _capturable_positions_in_reach(s, 0, full)
    assert len(expected) >= 1

    moves = _get_move_actions(s, 0)
    assert _move_positions(moves) == expected


def test_capture_move_gate_stochastic_second_get_legal_matches_first(monkeypatch):
    """Stochastic gate must not use luck_rng; repeated MOVE legals stay identical."""
    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "0.75")

    def boom(*_a, **_k):
        raise AssertionError("luck_rng.random must not gate capture moves (use hash trial)")

    s = _blank_state()
    pos_prop = _find_infantry_adjacent_to_enemy_property(s)
    assert pos_prop[0] is not None
    inf_pos, _ = pos_prop
    inf = _spawn(s, UnitType.INFANTRY, 0, inf_pos)
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))

    monkeypatch.setattr(s.luck_rng, "random", boom)
    moves1 = _get_move_actions(s, 0)
    moves2 = _get_move_actions(s, 0)
    assert moves1 == moves2


def test_capture_move_gate_stochastic_applies_when_trial_restricts(monkeypatch):
    import engine.action as act

    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "0.75")
    monkeypatch.setattr(
        act, "_stochastic_capture_gate_restrict", lambda *a, **k: True
    )

    s = _blank_state()
    pos_prop = _find_infantry_adjacent_to_enemy_property(s)
    assert pos_prop[0] is not None
    inf_pos, _ = pos_prop
    inf = _spawn(s, UnitType.INFANTRY, 0, inf_pos)
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))

    full = get_reachable_tiles(s, s.selected_unit)
    expected = _capturable_positions_in_reach(s, 0, full)
    assert len(expected) >= 1

    moves = _get_move_actions(s, 0)
    assert _move_positions(moves) == expected


def test_capture_move_gate_stochastic_skips_when_trial_passes(monkeypatch):
    import engine.action as act

    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "0.75")
    monkeypatch.setattr(
        act, "_stochastic_capture_gate_restrict", lambda *a, **k: False
    )

    s = _blank_state()
    pos_prop = _find_infantry_adjacent_to_enemy_property(s)
    assert pos_prop[0] is not None
    inf_pos, _ = pos_prop
    inf = _spawn(s, UnitType.INFANTRY, 0, inf_pos)
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))

    full = get_reachable_tiles(s, s.selected_unit)

    moves = _get_move_actions(s, 0)
    assert _move_positions(moves) == full


def test_capture_move_gate_on_no_capturable_in_reach_unrestricted(monkeypatch):
    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "1")

    s = _blank_state()
    tile = _find_infantry_tile_no_capturable_in_reach(s)
    assert tile is not None, "fixture map needs a tile with reachable area but no capturable props"
    inf = _spawn(s, UnitType.INFANTRY, 0, tile)
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))

    full = get_reachable_tiles(s, s.selected_unit)
    assert not _capturable_positions_in_reach(s, 0, full)

    moves = _get_move_actions(s, 0)
    assert _move_positions(moves) == full


def test_capture_move_gate_on_does_not_affect_tank(monkeypatch):
    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "1")

    s = _blank_state()
    pos_prop = _find_infantry_adjacent_to_enemy_property(s)
    assert pos_prop[0] is not None
    inf_pos, _ = pos_prop
    tank = _spawn(s, UnitType.TANK, 0, inf_pos)
    assert not UNIT_STATS[tank.unit_type].can_capture

    s.step(Action(ActionType.SELECT_UNIT, unit_pos=tank.pos))

    full = get_reachable_tiles(s, s.selected_unit)
    moves = _get_move_actions(s, 0)
    assert _move_positions(moves) == full
