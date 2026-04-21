"""END_TURN gating: loaded transports with cargo do not block END_TURN."""

from __future__ import annotations

from engine.action import ActionStage, ActionType, get_legal_actions
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH


def _blank_state(map_id: int = 123858, p0_co: int = 1, p1_co: int = 1):
    m = load_map(map_id, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, p0_co, p1_co, tier_name="T3")
    s.units[0] = []
    s.units[1] = []
    s.active_player = 0
    s.action_stage = ActionStage.SELECT
    s.selected_unit = None
    s.selected_move_pos = None
    return s


def _spawn(s, ut: UnitType, player: int, pos, hp: int = 100, unit_id: int = 0):
    st = UNIT_STATS[ut]
    u = Unit(
        ut,
        player,
        hp,
        st.max_ammo,
        st.max_fuel,
        pos,
        False,
        [],
        False,
        20,
        unit_id or len(s.units[0]) + len(s.units[1]) + 1,
    )
    s.units[player].append(u)
    return u


def _cargo_unit(ut: UnitType, player: int, pos, unit_id: int) -> Unit:
    st = UNIT_STATS[ut]
    return Unit(
        ut,
        player,
        100,
        st.max_ammo,
        st.max_fuel,
        pos,
        False,
        [],
        False,
        20,
        unit_id,
    )


def _find_non_property_ground_tile(s):
    """Empty tile that is not a property cell and is infantry-passable."""
    from engine.terrain import get_move_cost, MOVE_INF, INF_PASSABLE

    prop_cells = {(p.row, p.col) for p in s.properties}
    for r in range(s.map_data.height):
        for c in range(s.map_data.width):
            if (r, c) in prop_cells:
                continue
            tid = s.map_data.terrain[r][c]
            if get_move_cost(tid, MOVE_INF) >= INF_PASSABLE:
                continue
            if s.get_unit_at(r, c) is not None:
                continue
            return (r, c)
    return None


def _has_end_turn(legal):
    return any(a.action_type == ActionType.END_TURN for a in legal)


def test_infantry_off_property_blocks_end_turn():
    """One unmoved infantry not on a property — END_TURN must not be legal."""
    s = _blank_state()
    pos = _find_non_property_ground_tile(s)
    assert pos is not None, "fixture needs a non-property infantry tile"
    _spawn(s, UnitType.INFANTRY, 0, pos)
    legal = get_legal_actions(s)
    assert not _has_end_turn(legal), (
        "unmoved infantry should block END_TURN (no loaded-transport carve-out)"
    )


def test_loaded_apc_only_unmoved_allows_end_turn_and_keeps_select():
    """Only unmoved unit is an APC with cargo — END_TURN legal; APC still selectable."""
    s = _blank_state()
    pos = _find_non_property_ground_tile(s)
    assert pos is not None
    apc = _spawn(s, UnitType.APC, 0, pos, unit_id=501)
    cargo = _cargo_unit(UnitType.INFANTRY, 0, apc.pos, unit_id=502)
    apc.loaded_units.append(cargo)

    legal = get_legal_actions(s)
    assert _has_end_turn(legal), "loaded transport should not block END_TURN"
    selects = [
        a for a in legal
        if a.action_type == ActionType.SELECT_UNIT and a.unit_pos == apc.pos
    ]
    assert selects, "SELECT_UNIT for the APC must remain available"


def test_empty_apc_blocks_end_turn():
    """Unmoved APC with no cargo — carve-out does not apply; END_TURN blocked."""
    s = _blank_state()
    pos = _find_non_property_ground_tile(s)
    assert pos is not None
    _spawn(s, UnitType.APC, 0, pos, unit_id=601)
    legal = get_legal_actions(s)
    assert not _has_end_turn(legal), (
        "empty transport must still block END_TURN until it acts or loads cargo"
    )
