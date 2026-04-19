"""
Runtime validation for Sami capture: D2D vs COP vs SCOP (prints scenario outcomes).
"""
from __future__ import annotations

from engine.action import Action, ActionType, ActionStage
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS

PLAIN = 1
NEUTRAL_CITY = 34


def _build_map(properties: list[PropertyState], terrain: list[list[int]]) -> MapData:
    return MapData(
        map_id=999_997,
        name="validate_sami_capture",
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


def _neutral_city(row: int, col: int) -> PropertyState:
    return PropertyState(
        terrain_id=NEUTRAL_CITY,
        row=row,
        col=col,
        owner=None,
        capture_points=20,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=False,
        is_airport=False,
        is_port=False,
    )


def _make_unit(state: GameState, unit_type: UnitType, player: int, pos: tuple[int, int], hp: int = 100) -> Unit:
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


def _terrain_with_city() -> list[list[int]]:
    t = [[PLAIN] * 5 for _ in range(5)]
    t[2][2] = NEUTRAL_CITY
    return t


def _select_move_capture(state: GameState, unit: Unit, dest: tuple[int, int]) -> None:
    state.action_stage = ActionStage.SELECT
    state.selected_unit = None
    state.selected_move_pos = None
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos))
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=dest))
    state.step(Action(ActionType.CAPTURE, unit_pos=unit.pos, move_pos=dest))


def _fresh_state(p0_co: int) -> GameState:
    md = _build_map([_neutral_city(2, 2)], _terrain_with_city())
    st = make_initial_state(md, p0_co, 1, starting_funds=0, tier_name="T2")
    st.units = {0: [], 1: []}
    st.active_player = 0
    return st


def main() -> None:
    scenarios = []

    # 1) Sami D2D (no power): full HP infantry — AWBW table gives 15 CP / action at 10 HP
    st = _fresh_state(8)
    inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))
    _select_move_capture(st, inf, (2, 2))
    prop = st.get_property_at(2, 2)
    scenarios.append(("sami_d2d_full_hp_inf", prop.owner, prop.capture_points if prop else None))

    # 2) Sami COP: capture matches D2D (COP does not add extra capture in AWBW)
    st = _fresh_state(8)
    st.co_states[0].cop_active = True
    inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))
    _select_move_capture(st, inf, (2, 2))
    prop = st.get_property_at(2, 2)
    scenarios.append(("sami_cop_full_hp_inf", prop.owner, prop.capture_points if prop else None))

    # 3) Sami SCOP: instant capture
    st = _fresh_state(8)
    st.co_states[0].scop_active = True
    inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))
    _select_move_capture(st, inf, (2, 2))
    prop = st.get_property_at(2, 2)
    scenarios.append(("sami_scop_full_hp_inf", prop.owner, prop.capture_points if prop else None))

    # 4) Andy baseline: single-rate capture (10 CP one step)
    st = _fresh_state(1)
    inf = _make_unit(st, UnitType.INFANTRY, 0, (2, 3))
    _select_move_capture(st, inf, (2, 2))
    prop = st.get_property_at(2, 2)
    scenarios.append(("andy_d2d_full_hp_inf", prop.owner, prop.capture_points if prop else None))

    # 5) Sami D2D mech (same path as infantry in engine)
    st = _fresh_state(8)
    mech = _make_unit(st, UnitType.MECH, 0, (2, 3))
    _select_move_capture(st, mech, (2, 2))
    prop = st.get_property_at(2, 2)
    scenarios.append(("sami_d2d_mech", prop.owner, prop.capture_points if prop else None))

    for name, owner, cp in scenarios:
        print(f"{name}: owner={owner!r} capture_points={cp!r}")


if __name__ == "__main__":
    main()
