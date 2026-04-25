"""Eagle (co_id 10) — AWBW wiki https://awbw.fandom.com/wiki/Eagle

Pins D2D / COP / SCOP stat stacking (with universal SCOPB on powers) and
Lightning Strike ``moved`` refresh + SCOP build / unload carve-outs.
"""
from __future__ import annotations

from engine.action import Action, ActionStage, ActionType, get_legal_actions
from engine.co import make_co_state, make_co_state_safe
from engine.combat import calculate_damage
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.terrain import get_terrain
from engine.unit import Unit, UnitType, UNIT_STATS

PLAIN = 1
EAGLE_CO_ID = 10
ANDY_CO_ID = 1
ROAD_TID = 15


def _make_state(
    *,
    p0_co: int = EAGLE_CO_ID,
    p1_co: int = ANDY_CO_ID,
    active_player: int = 0,
) -> GameState:
    terrain = [[PLAIN] * 12 for _ in range(12)]
    md = MapData(
        map_id=999_996,
        name="eagle_probe",
        map_type="std",
        terrain=terrain,
        height=12,
        width=12,
        cap_limit=999,
        unit_limit=999,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[50_000, 0],
        co_states=[make_co_state_safe(p0_co), make_co_state_safe(p1_co)],
        properties=[],
        turn=1,
        active_player=active_player,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T1",
        full_trace=[],
        seam_hp={},
    )


_NEXT_UID = [33000]


def _spawn(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    moved: bool = False,
) -> Unit:
    stats = UNIT_STATS[ut]
    _NEXT_UID[0] += 1
    u = Unit(
        unit_type=ut,
        player=player,
        hp=100,
        ammo=stats.max_ammo,
        fuel=stats.max_fuel,
        pos=pos,
        moved=moved,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=_NEXT_UID[0],
    )
    state.units[player].append(u)
    return u


def _fire_scop(state: GameState) -> None:
    co = state.co_states[state.active_player]
    co.power_bar = co._scop_threshold
    state.step(Action(ActionType.ACTIVATE_SCOP))


def test_lightning_strike_clears_moved_for_non_foot_only() -> None:
    state = _make_state()
    tank = _spawn(state, UnitType.TANK, 0, (1, 1), moved=True)
    inf = _spawn(state, UnitType.INFANTRY, 0, (2, 2), moved=True)
    _fire_scop(state)
    assert tank.moved is False
    assert inf.moved is True


def test_lightning_strike_vehicle_selectable_under_step_gate() -> None:
    """RL legality: after SCOP, non–inf/mech with moved cleared must be selectable.

    Oracle zip replay bypasses ``get_legal_actions`` (``oracle_mode=True``).
    This guards the default STEP-GATE path so Eagle double-activation cannot
    drift without failing ``pytest`` even when ``desync_audit`` stays green.
    """
    state = _make_state()
    _spawn(state, UnitType.TANK, 0, (1, 1), moved=True)
    _fire_scop(state)
    assert state.action_stage == ActionStage.SELECT
    legal = get_legal_actions(state)
    assert Action(ActionType.SELECT_UNIT, unit_pos=(1, 1)) in legal


def _state_with_p0_base_eagle() -> GameState:
    terrain = [[1, 39]]
    prop = PropertyState(
        terrain_id=39,
        row=0,
        col=1,
        owner=0,
        capture_points=20,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=True,
        is_airport=False,
        is_port=False,
    )
    md = MapData(
        map_id=999_995,
        name="eagle_build",
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
        predeployed_specs=[],
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[30_000, 0],
        co_states=[make_co_state_safe(EAGLE_CO_ID), make_co_state_safe(ANDY_CO_ID)],
        properties=md.properties,
        turn=1,
        active_player=0,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T1",
        full_trace=[],
        seam_hp={},
    )


def test_lightning_strike_non_foot_build_can_move_same_turn() -> None:
    st = _state_with_p0_base_eagle()
    _fire_scop(st)
    st.step(Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.TANK))
    assert len(st.units[0]) == 1
    assert st.units[0][0].moved is False


def test_lightning_strike_infantry_build_still_exhausted() -> None:
    st = _state_with_p0_base_eagle()
    _fire_scop(st)
    st.step(Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.INFANTRY))
    assert len(st.units[0]) == 1
    assert st.units[0][0].moved is True


def test_eagle_build_without_scop_still_exhausted() -> None:
    st = _state_with_p0_base_eagle()
    st.step(Action(ActionType.BUILD, move_pos=(0, 1), unit_type=UnitType.TANK))
    assert st.units[0][0].moved is True


def test_eagle_d2d_air_and_naval_modifiers_match_wiki_table() -> None:
    """Wiki D2D: air 115/110 vs 100/100; naval atk 70/100 (Battleship row)."""
    eagle = make_co_state(EAGLE_CO_ID)
    assert eagle.total_atk("air") == 115
    assert eagle.total_def("air") == 110
    assert eagle.total_atk("naval") == 70
    assert eagle.total_def("naval") == 100


def test_eagle_cop_air_totals_match_wiki_table_130_130() -> None:
    """Wiki COP column: Fighter 130/130 (SCOPB +10 each + D2D + power air)."""
    eagle = make_co_state(EAGLE_CO_ID)
    eagle.cop_active = True
    assert eagle.total_atk("air") == 130
    assert eagle.total_def("air") == 130
    assert eagle.total_atk("infantry") == 110
    assert eagle.total_def("infantry") == 110


def test_eagle_scop_air_totals_match_wiki_table() -> None:
    """Wiki SCOP column matches COP for air (130/130 Fighter)."""
    eagle = make_co_state(EAGLE_CO_ID)
    eagle.scop_active = True
    assert eagle.total_atk("air") == 130
    assert eagle.total_def("air") == 130


def test_eagle_d2d_naval_attacker_reduces_damage_vs_neutral() -> None:
    """Naval -30% ATK D2D: Eagle Battleship should deal less than Andy."""
    road = get_terrain(ROAD_TID)
    atk_e = Unit(
        unit_type=UnitType.BATTLESHIP,
        player=0,
        hp=100,
        ammo=6,
        fuel=99,
        pos=(0, 0),
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
    )
    defn = Unit(
        unit_type=UnitType.BATTLESHIP,
        player=1,
        hp=100,
        ammo=6,
        fuel=99,
        pos=(0, 1),
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
    )
    eagle = make_co_state(EAGLE_CO_ID)
    andy = make_co_state(ANDY_CO_ID)
    d_eagle = calculate_damage(atk_e, defn, road, road, eagle, andy, luck_roll=0)
    d_andy = calculate_damage(atk_e, defn, road, road, andy, andy, luck_roll=0)
    assert d_eagle is not None and d_andy is not None
    assert d_eagle < d_andy
