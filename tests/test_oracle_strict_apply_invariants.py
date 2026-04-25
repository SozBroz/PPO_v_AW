"""Phase 11B + 11M Wave 2 — ``oracle_strict`` invariants for selected ``_apply_*``.

Phase 11B: BUILD / JOIN / REPAIR. Phase 11M Wave 2: LOAD (S10O-09), UNLOAD (S10O-14..17).

Default ``oracle_zip_replay`` uses ``oracle_mode=True`` and ``oracle_strict=False``;
these tests cover the strict audit lane and the REPAIR guard default-path fix.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState, IllegalActionError
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS

# Terrain IDs (engine/terrain.py)
_PLAIN = 1
_SEA = 28


def _tiny_transport_map_state(terrain: list[list[int]]) -> GameState:
    h, w = len(terrain), len(terrain[0])
    md = MapData(
        map_id=990_011,
        name="oracle-strict-transport",
        map_type="std",
        terrain=[row[:] for row in terrain],
        height=h,
        width=w,
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
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


def _spawn_with_loaded(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    unit_id: int = 5001,
    moved: bool = False,
    loaded_units: list[Unit] | None = None,
) -> Unit:
    stats = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=100,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=moved,
        loaded_units=loaded_units or [],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


def _infantry_cargo_at(pos: tuple[int, int], unit_id: int) -> Unit:
    st = UNIT_STATS[UnitType.INFANTRY]
    return Unit(
        unit_type=UnitType.INFANTRY,
        player=0,
        hp=100,
        ammo=st.max_ammo,
        fuel=st.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )


def _minimal_build_state(
    *,
    active_player: int = 0,
    factory_owner: int | None = 0,
    funds_p0: int = 10_000,
) -> GameState:
    terrain = [[1, 35]]  # plain, neutral base
    prop = PropertyState(
        terrain_id=35,
        row=0,
        col=1,
        owner=factory_owner,
        capture_points=20,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=True,
        is_airport=False,
        is_port=False,
    )
    map_data = MapData(
        map_id=0,
        name="oracle-strict-build",
        map_type="std",
        terrain=terrain,
        height=1,
        width=2,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[prop],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=map_data,
        units={0: [], 1: []},
        funds=[funds_p0, 10_000],
        co_states=[make_co_state_safe(0), make_co_state_safe(0)],
        properties=map_data.properties,
        turn=1,
        active_player=active_player,
        action_stage=ActionStage.SELECT,
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
    unit_id: int = 5001,
    moved: bool = False,
) -> Unit:
    stats = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=100,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=moved,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


def _empty_join_state() -> GameState:
    terrain = [[1, 1, 1]]
    md = MapData(
        map_id=1,
        name="join-strict",
        map_type="std",
        terrain=terrain,
        height=1,
        width=3,
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
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


# ---------------------------------------------------------------------------
# BUILD (S10O-18..23) — sample branch: insufficient funds (S10O-22)
# ---------------------------------------------------------------------------


def test_build_insufficient_funds_oracle_strict_raises():
    state = _minimal_build_state(funds_p0=0)
    act = Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.INFANTRY)
    with pytest.raises(IllegalActionError, match="BUILD: insufficient funds"):
        state.step(act, oracle_mode=True, oracle_strict=True)


def test_build_insufficient_funds_oracle_strict_false_no_raise():
    state = _minimal_build_state(funds_p0=0)
    before = state.funds[0]
    act = Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.INFANTRY)
    state.step(act, oracle_mode=True, oracle_strict=False)
    assert state.funds[0] == before
    assert len(state.units[0]) == 0


# ---------------------------------------------------------------------------
# JOIN (S10O-10)
# ---------------------------------------------------------------------------


def test_join_no_partner_oracle_strict_raises():
    state = _empty_join_state()
    _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=7001)
    act = Action(
        ActionType.JOIN,
        unit_pos=(0, 0),
        move_pos=(0, 2),
    )
    with pytest.raises(IllegalActionError, match="JOIN: no merge partner"):
        state.step(act, oracle_mode=True, oracle_strict=True)


def test_join_no_partner_oracle_strict_false_no_raise():
    state = _empty_join_state()
    inf = _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=7002)
    act = Action(
        ActionType.JOIN,
        unit_pos=(0, 0),
        move_pos=(0, 2),
    )
    state.step(act, oracle_mode=True, oracle_strict=False)
    assert inf in state.units[0]
    assert len(state.units[0]) == 1


# ---------------------------------------------------------------------------
# REPAIR (S10O-04) — Black Boat guard at unit_pos
# ---------------------------------------------------------------------------


def test_repair_non_boat_oracle_strict_raises():
    state = _empty_join_state()
    _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=8001)
    act = Action(
        ActionType.REPAIR,
        unit_pos=(0, 0),
        move_pos=(0, 0),
        target_pos=(0, 1),
    )
    with pytest.raises(IllegalActionError, match="REPAIR: not a Black Boat"):
        state.step(act, oracle_mode=True, oracle_strict=True)


def test_repair_non_boat_oracle_strict_false_finishes_like_finish_action():
    state = _empty_join_state()
    inf = _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=8002, moved=False)
    state.action_stage = ActionStage.ACTION
    state.selected_unit = inf
    state.selected_move_pos = (0, 0)
    act = Action(
        ActionType.REPAIR,
        unit_pos=(0, 0),
        move_pos=(0, 0),
        target_pos=(0, 1),
    )
    with patch.object(state, "_finish_action", wraps=state._finish_action) as fin:
        state.step(act, oracle_mode=True, oracle_strict=False)
        fin.assert_called_once_with(inf)
    assert inf.moved is True
    assert state.action_stage == ActionStage.SELECT
    assert state.selected_unit is None


def test_repair_missing_unit_oracle_strict_false_clears_stage():
    state = _empty_join_state()
    state.action_stage = ActionStage.ACTION
    state.selected_unit = None
    state.selected_move_pos = (0, 0)
    act = Action(
        ActionType.REPAIR,
        unit_pos=(0, 0),
        move_pos=(0, 0),
        target_pos=(0, 1),
    )
    state.step(act, oracle_mode=True, oracle_strict=False)
    assert state.action_stage == ActionStage.SELECT
    assert state.selected_unit is None
    assert state.selected_move_pos is None


# ---------------------------------------------------------------------------
# Phase 11M Wave 2 — LOAD (S10O-09), UNLOAD (S10O-14..17)
# ---------------------------------------------------------------------------


def test_apply_load_missing_mover_or_transport_oracle_strict_raises():
    state = _tiny_transport_map_state([[_PLAIN, _PLAIN]])
    _spawn(state, UnitType.APC, 0, (0, 1), unit_id=9101)
    act = Action(
        ActionType.LOAD,
        unit_pos=(0, 0),
        move_pos=(0, 1),
    )
    with pytest.raises(IllegalActionError, match="_apply_load: mover or transport missing"):
        state.step(act, oracle_mode=True, oracle_strict=True)


def test_apply_load_missing_mover_or_transport_oracle_strict_false_no_raise():
    state = _tiny_transport_map_state([[_PLAIN, _PLAIN]])
    apc = _spawn(state, UnitType.APC, 0, (0, 1), unit_id=9102)
    before = list(state.units[0])
    act = Action(
        ActionType.LOAD,
        unit_pos=(0, 0),
        move_pos=(0, 1),
    )
    state.step(act, oracle_mode=True, oracle_strict=False)
    assert state.units[0] == before
    assert apc.loaded_units == []


def test_apply_unload_drop_not_adjacent_after_move_oracle_strict_raises():
    terrain = [[_PLAIN] * 6]
    state = _tiny_transport_map_state(terrain)
    cargo = _infantry_cargo_at((0, 1), 9201)
    apc = _spawn_with_loaded(
        state, UnitType.APC, 0, (0, 1), unit_id=9200, loaded_units=[cargo]
    )
    act = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 1),
        move_pos=(0, 0),
        target_pos=(0, 3),
        unit_type=UnitType.INFANTRY,
    )
    with pytest.raises(
        IllegalActionError,
        match="drop tile not orthogonally adjacent",
    ):
        state.step(act, oracle_mode=True, oracle_strict=True)


def test_apply_unload_drop_not_adjacent_after_move_oracle_strict_false_partial_move():
    terrain = [[_PLAIN] * 6]
    state = _tiny_transport_map_state(terrain)
    cargo = _infantry_cargo_at((0, 1), 9301)
    apc = _spawn_with_loaded(
        state, UnitType.APC, 0, (0, 1), unit_id=9300, loaded_units=[cargo]
    )
    act = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 1),
        move_pos=(0, 0),
        target_pos=(0, 3),
        unit_type=UnitType.INFANTRY,
    )
    state.step(act, oracle_mode=True, oracle_strict=False)
    assert apc.pos == (0, 0)
    assert len(apc.loaded_units) == 1


def test_apply_unload_drop_out_of_bounds_oracle_strict_raises():
    state = _tiny_transport_map_state([[_PLAIN, _PLAIN]])
    cargo = _infantry_cargo_at((0, 1), 9401)
    _spawn_with_loaded(
        state, UnitType.APC, 0, (0, 1), unit_id=9400, loaded_units=[cargo]
    )
    act = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 1),
        move_pos=(0, 1),
        target_pos=(-1, 1),
        unit_type=UnitType.INFANTRY,
    )
    with pytest.raises(IllegalActionError, match="drop position out of bounds"):
        state.step(act, oracle_mode=True, oracle_strict=True)


def test_apply_unload_drop_out_of_bounds_oracle_strict_false_no_raise():
    state = _tiny_transport_map_state([[_PLAIN, _PLAIN]])
    cargo = _infantry_cargo_at((0, 1), 9501)
    apc = _spawn_with_loaded(
        state, UnitType.APC, 0, (0, 1), unit_id=9500, loaded_units=[cargo]
    )
    act = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 1),
        move_pos=(0, 1),
        target_pos=(-1, 1),
        unit_type=UnitType.INFANTRY,
    )
    state.step(act, oracle_mode=True, oracle_strict=False)
    assert apc.pos == (0, 1)
    assert len(apc.loaded_units) == 1


def test_apply_unload_drop_tile_occupied_oracle_strict_raises():
    terrain = [[_PLAIN, _PLAIN, _PLAIN]]
    state = _tiny_transport_map_state(terrain)
    cargo = _infantry_cargo_at((0, 1), 9601)
    _spawn_with_loaded(
        state, UnitType.APC, 0, (0, 1), unit_id=9600, loaded_units=[cargo]
    )
    _spawn(state, UnitType.INFANTRY, 0, (0, 2), unit_id=9602)
    act = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 1),
        move_pos=(0, 1),
        target_pos=(0, 2),
        unit_type=UnitType.INFANTRY,
    )
    with pytest.raises(IllegalActionError, match="drop tile occupied"):
        state.step(act, oracle_mode=True, oracle_strict=True)


def test_apply_unload_drop_tile_occupied_oracle_strict_false_no_raise():
    terrain = [[_PLAIN, _PLAIN, _PLAIN]]
    state = _tiny_transport_map_state(terrain)
    cargo = _infantry_cargo_at((0, 1), 9701)
    apc = _spawn_with_loaded(
        state, UnitType.APC, 0, (0, 1), unit_id=9700, loaded_units=[cargo]
    )
    blocker = _spawn(state, UnitType.INFANTRY, 0, (0, 2), unit_id=9702)
    act = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 1),
        move_pos=(0, 1),
        target_pos=(0, 2),
        unit_type=UnitType.INFANTRY,
    )
    state.step(act, oracle_mode=True, oracle_strict=False)
    assert apc.pos == (0, 1)
    assert len(apc.loaded_units) == 1
    assert blocker in state.units[0]


def test_apply_unload_drop_terrain_impassable_for_cargo_oracle_strict_raises():
    terrain = [[_PLAIN, _SEA]]
    state = _tiny_transport_map_state(terrain)
    cargo = _infantry_cargo_at((0, 0), 9801)
    _spawn_with_loaded(
        state, UnitType.APC, 0, (0, 0), unit_id=9800, loaded_units=[cargo]
    )
    act = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 0),
        move_pos=(0, 0),
        target_pos=(0, 1),
        unit_type=UnitType.INFANTRY,
    )
    with pytest.raises(IllegalActionError, match="drop terrain impassable"):
        state.step(act, oracle_mode=True, oracle_strict=True)


def test_apply_unload_drop_terrain_impassable_oracle_strict_false_no_raise():
    terrain = [[_PLAIN, _SEA]]
    state = _tiny_transport_map_state(terrain)
    cargo = _infantry_cargo_at((0, 0), 9901)
    apc = _spawn_with_loaded(
        state, UnitType.APC, 0, (0, 0), unit_id=9900, loaded_units=[cargo]
    )
    act = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 0),
        move_pos=(0, 0),
        target_pos=(0, 1),
        unit_type=UnitType.INFANTRY,
    )
    state.step(act, oracle_mode=True, oracle_strict=False)
    assert apc.pos == (0, 0)
    assert len(apc.loaded_units) == 1
