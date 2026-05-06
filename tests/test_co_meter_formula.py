"""CO meter credit from combat (display-bucket AWBW formula)."""
from __future__ import annotations

from engine.action import ActionStage
from engine.game import GameState
from engine.co import make_co_state_safe
from engine.unit import Unit, UnitType, UNIT_STATS
from engine.map_loader import MapData


def _blank_state() -> GameState:
    """Create a minimal GameState with Andy (co_id=1) for both seats."""
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
    """User example: AA kills fresh infantry (100 internal HP)."""
    state = _blank_state()
    aa = _unit(UnitType.ANTI_AIR, player=0, hp=100, uid=501)
    inf = _unit(UnitType.INFANTRY, player=1, hp=100, uid=502)
    # 100 internal HP lost (full kill = 100 internal HP)
    state._apply_co_meter_from_internal_hp_lost(aa, inf, 100)
    # P0 (striker, AA=8000): 100 × 8000 ÷ 180 = 4444
    # P1 (victim, Infantry=1000): 100 × 1000 ÷ 90 = 1111
    assert state.co_states[0].power_bar == 4444
    assert state.co_states[1].power_bar == 1111


def test_meter_exchange_copter_chunks() -> None:
    """User example swing: 70 internal HP lost (7 display HP)."""
    state = _blank_state()
    bc = _unit(UnitType.B_COPTER, player=0, hp=100, uid=701)
    bc2 = _unit(UnitType.B_COPTER, player=1, hp=91, uid=702)
    # B_COPTER cost = 9000, 70 internal HP lost (7 display HP)
    # P0 (striker, 9000): 70 × 9000 ÷ 180 = 3500
    # P1 (victim, 9000): 70 × 9000 ÷ 90 = 7000 (no cap at SCOP threshold)
    state._apply_co_meter_from_internal_hp_lost(bc, bc2, 70)
    assert state.co_states[0].power_bar == 3500
    assert state.co_states[1].power_bar == 7000  # Not capped at SCOP threshold


def test_meter尹ch_recon_split_main_and_counter() -> None:
    """User scenario 3: main hit Δ=50 internal HP on mech, counter Δ=20 on recon."""
    state = _blank_state()
    # Recon cost = 4000, Mech cost = 3000
    recon = _unit(UnitType.RECON, player=1, hp=100, uid=901)
    mech = _unit(UnitType.MECH, player=0, hp=90, uid=902)
    # First hit: recon (P1, 4000) hits mech (P0, 3000) for 50 internal HP
    # P0 (victim, Mech=3000): 50 × 3000 ÷ 90 = 1666.67 → 1667 (rounded half up)
    # P1 (striker, recon=4000): 50 × 4000 ÷ 180 = 1111
    state._apply_co_meter_from_internal_hp_lost(recon, mech, 50)
    assert state.co_states[0].power_bar == 1667  # 1666.67 rounded half up
    assert state.co_states[1].power_bar == 1111
    # Counterattack: mech (P0, 3000) hits recon (P1, 4000) for 20 internal HP
    # P1 (victim, recon=4000): 20 × 4000 ÷ 90 = 888
    # P0 (striker, mech=3000): 20 × 3000 ÷ 180 = 333
    state._apply_co_meter_from_internal_hp_lost(mech, recon, 20)
    assert state.co_states[0].power_bar == 2000  # 1667 + 333
    assert state.co_states[1].power_bar == 2000  # 1111 + 889


def test_meter_skips_seat_under_active_power() -> None:
    """CO meter still charges even when COP is active (current implementation)."""
    state = _blank_state()
    state.co_states[0].cop_active = True
    aa = _unit(UnitType.ANTI_AIR, player=0, hp=100, uid=1)
    inf = _unit(UnitType.INFANTRY, player=1, hp=100, uid=2)
    # Even with COP active, meter still charges
    # 100 internal HP lost (full kill)
    # P0 (striker, AA=8000): 100 × 8000 ÷ 180 = 4444
    # P1 (victim, Infantry=1000): 100 × 1000 ÷ 90 = 1111
    state._apply_co_meter_from_internal_hp_lost(aa, inf, 100)
    assert state.co_states[0].power_bar == 4444  # Active but still charges
    assert state.co_states[1].power_bar == 1111


def test_activate_cop_subtracts_threshold_retains_remainder() -> None:
    """COP activation subtracts threshold, retains remainder (like AWBW)."""
    state = _blank_state()
    # Andy: threshold=6 stars = 54000
    state.co_states[0].power_bar = 60000  # 6.66 stars
    state.co_states[0].cop_active = False
    # Simulate COP activation (subtract 6 * 9000 = 54000)
    state.co_states[0].power_bar -= 6 * 9000
    assert state.co_states[0].power_bar == 6000  # 6000 is 0.66 stars retained
