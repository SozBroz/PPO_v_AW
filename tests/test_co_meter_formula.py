"""CO meter credit from combat (display-bucket AWBW formula)."""
from __future__ import annotations

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import Unit, UnitType, UNIT_STATS


def _blank_state() -> GameState:
    md = MapData(
        width=5,
        height=5,
        terrain=[[1] * 5 for _ in range(5)],
        properties=[],
        unit_limit=50,
        map_id=0,
        name="blank",
        map_type="normal",
        cap_limit=0,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(1), make_co_state_safe(1)],
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


def _unit(tp: UnitType, player: int, *, hp: int, uid: int) -> Unit:
    stats = UNIT_STATS[tp]
    return Unit(
        unit_id=uid,
        unit_type=tp,
        player=player,
        pos=(0, 0),
        hp=hp,
        ammo=stats.max_ammo,
        fuel=stats.max_fuel,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=0,
        is_stunned=False,
    )


def test_meter_full_kill_infantry_vs_aa() -> None:
    """AA kills infantry: 10 display bars; striker half uses victim (inf) cost."""
    state = _blank_state()
    aa = _unit(UnitType.ANTI_AIR, player=0, hp=100, uid=501)
    inf = _unit(UnitType.INFANTRY, player=1, hp=100, uid=502)
    state._apply_co_meter_from_display_buckets_lost(aa, inf, 10)
    assert state.co_states[0].power_bar == 500   # 10 × 1000 ÷ 20
    assert state.co_states[1].power_bar == 1000  # 10 × 1000 ÷ 10


def test_meter_exchange_copter_chunks() -> None:
    """7 display bars lost on B-Copter."""
    state = _blank_state()
    bc = _unit(UnitType.B_COPTER, player=0, hp=100, uid=701)
    bc2 = _unit(UnitType.B_COPTER, player=1, hp=100, uid=702)
    state._apply_co_meter_from_display_buckets_lost(bc, bc2, 7)
    assert state.co_states[0].power_bar == 3150
    assert state.co_states[1].power_bar == 6300


def test_meter_recon_mech_split_main_and_counter() -> None:
    """Recon hits mech 6 bars; mech counters 2 bars on recon."""
    state = _blank_state()
    recon = _unit(UnitType.RECON, player=1, hp=100, uid=901)
    mech = _unit(UnitType.MECH, player=0, hp=90, uid=902)
    state._apply_co_meter_from_display_buckets_lost(recon, mech, 6)
    assert state.co_states[0].power_bar == 1800
    assert state.co_states[1].power_bar == 900
    state._apply_co_meter_from_display_buckets_lost(mech, recon, 2)
    assert state.co_states[0].power_bar == 2200
    assert state.co_states[1].power_bar == 1700


def test_meter_infantry_vs_aa_exchange() -> None:
    """P1 inf 8→0, P0 AA 10→9: AWBW symmetric 1200 each."""
    state = _blank_state()
    inf = _unit(UnitType.INFANTRY, player=1, hp=80, uid=5)
    aa = _unit(UnitType.ANTI_AIR, player=0, hp=100, uid=18)
    state._apply_co_meter_from_display_buckets_lost(inf, aa, 1)
    state._apply_co_meter_from_display_buckets_lost(aa, inf, 8)
    assert state.co_states[0].power_bar == 1200
    assert state.co_states[1].power_bar == 1200


def test_meter_aa_vs_tank_exchange() -> None:
    """P0 AA hits P1 tank 5→2 (3 bars); tank counters AA 9→8 (1 bar)."""
    state = _blank_state()
    aa = _unit(UnitType.ANTI_AIR, player=0, hp=90, uid=1)
    tank = _unit(UnitType.TANK, player=1, hp=50, uid=2)
    state._apply_co_meter_from_display_buckets_lost(aa, tank, 3)
    state._apply_co_meter_from_display_buckets_lost(tank, aa, 1)
    assert state.co_states[0].power_bar == 1850  # 1050 dealt + 800 received
    assert state.co_states[1].power_bar == 2500  # 2100 received + 400 dealt


def test_meter_aa_vs_tank_via_apply_attack() -> None:
    """Full _apply_attack path with oracle-pinned display deltas."""
    state = _blank_state()
    state.map_data = MapData(
        map_id=0,
        name="aa_tank",
        map_type="std",
        terrain=[[3, 3], [3, 3]],
        height=2,
        width=2,
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
    aa = _unit(UnitType.ANTI_AIR, player=0, hp=90, uid=1)
    aa.pos = (0, 0)
    tank = _unit(UnitType.TANK, player=1, hp=50, uid=2)
    tank.pos = (0, 1)
    state.units[0] = [aa]
    state.units[1] = [tank]
    state.active_player = 0
    state.action_stage = ActionStage.ACTION
    state.selected_unit = aa
    state.selected_move_pos = (0, 0)
    state._oracle_combat_damage_override = (30, 10)
    state._apply_attack(
        Action(ActionType.ATTACK, unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 1))
    )
    assert aa.display_hp == 8
    assert tank.display_hp == 2
    assert state.co_states[0].power_bar == 1850
    assert state.co_states[1].power_bar == 2500


def test_meter_no_credit_without_display_tick() -> None:
    state = _blank_state()
    tank = _unit(UnitType.TANK, player=0, hp=100, uid=1)
    other = _unit(UnitType.TANK, player=1, hp=100, uid=2)
    state._apply_co_meter_from_display_buckets_lost(tank, other, 0)
    assert state.co_states[0].power_bar == 0
    assert state.co_states[1].power_bar == 0


def test_meter_skips_charge_while_power_active() -> None:
    """AWBW: no meter gain for a seat while its COP/SCOP is active."""
    state = _blank_state()
    state.co_states[0].cop_active = True
    aa = _unit(UnitType.ANTI_AIR, player=0, hp=100, uid=1)
    inf = _unit(UnitType.INFANTRY, player=1, hp=100, uid=2)
    state._apply_co_meter_from_display_buckets_lost(aa, inf, 10)
    assert state.co_states[0].power_bar == 0
    assert state.co_states[1].power_bar == 1000


def test_power_active_until_next_turn_start() -> None:
    """COP persists through opponent turn; clears when that seat acts again."""
    state = _blank_state()
    state.active_player = 0
    state.co_states[0].cop_active = True

    state._end_turn()  # P0 ends → P1's turn
    assert state.active_player == 1
    assert state.co_states[0].cop_active is True

    state._end_turn()  # P1 ends → P0's next turn
    assert state.active_player == 0
    assert state.co_states[0].cop_active is False
    assert state.co_states[0].scop_active is False


def test_activate_cop_subtracts_threshold_retains_remainder() -> None:
    state = _blank_state()
    state.co_states[0].power_bar = 60000
    state.co_states[0].power_bar -= 3 * 9000
    assert state.co_states[0].power_bar == 33000
