"""Phase 11J-SASHA-WARBONDS-SHIP — Sasha SCOP "War Bonds" funds payout.

AWBW canon (Tier 1, AWBW CO Chart Sasha row):
  *"War Bonds — Returns 50% of damage dealt as funds (subject to a 9HP cap)."*
  https://awbw.amarriner.com/co.php

Engine model (see engine/co.py COState war_bonds_active /
pending_war_bonds_funds and engine/game.py _apply_war_bonds_payout):

* Per-attack payout = ``min(display_hp_loss, 9) * unit_cost(target) // 20``
  (display_hp_loss = ceil(internal_hp/10)). All AWBW unit costs are
  multiples of 1000, so cost//20 is always integer.
* Hybrid crediting (Phase 11J-L1-WAVE-2-SHIP):
  - **Own SCOP-turn attacks** (damage_dealer == active_player):
    credited IMMEDIATELY to Sasha's treasury so the in-turn builds
    can spend the bonds. Empirical: gids 1624082, 1626284, 1628953,
    1634267, 1634893 all show Sasha SCOP at envelope action [0]
    followed by ≥4 attacks and builds totalling more than pre-SCOP
    funds — only real-time crediting explains why PHP allows the
    builds.
  - **Counter-attacks during opp's intervening turn**
    (damage_dealer != active_player): accumulated into
    ``pending_war_bonds_funds`` and credited at the END of opp's
    turn. Preserves the 1624082 env-22 −200g empirical anchor and
    avoids the 23/100 mid-turn spending-power regression observed
    when opp-turn counter-attacks were also credited in real time
    (docs/oracle_exception_audit/phase11j_sasha_warbonds_ship.md).

Empirical anchor: game ``1624082`` env 22 — Δfunds = exactly −200g
locked in from env 22 onward (recon: phase11j_funds_deep.md §5.1).
"""
from __future__ import annotations

import pytest

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import Unit, UnitType, UNIT_STATS

PLAIN = 1

SASHA_CO_ID = 19
ANDY_CO_ID = 1


def _make_state(*, width: int = 5, height: int = 5,
                p0_co_id: int = SASHA_CO_ID,
                p1_co_id: int = ANDY_CO_ID) -> GameState:
    md = MapData(
        map_id=0, name="warbonds", map_type="std",
        terrain=[[PLAIN] * width for _ in range(height)],
        height=height, width=width,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=[],
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(p0_co_id), make_co_state_safe(p1_co_id)],
        properties=[],
        turn=1,
        active_player=0,
        action_stage=ActionStage.ACTION,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


def _spawn(state: GameState, ut: UnitType, player: int,
           pos: tuple[int, int], *, unit_id: int, hp: int = 100,
           ammo: int | None = None) -> Unit:
    stats = UNIT_STATS[ut]
    if ammo is None:
        ammo = stats.max_ammo if stats.max_ammo > 0 else 0
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
        ammo=ammo,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


def _attack(state: GameState, attacker: Unit, target_pos: tuple[int, int],
            *, override_dmg: int | None = None,
            override_counter: int | None = None) -> None:
    """Pin oracle damage (if requested) and apply the ATTACK action."""
    state.selected_unit = attacker
    state.selected_move_pos = attacker.pos
    if override_dmg is not None or override_counter is not None:
        state._oracle_combat_damage_override = (override_dmg, override_counter)
    state._apply_attack(
        Action(
            ActionType.ATTACK,
            unit_pos=attacker.pos,
            move_pos=attacker.pos,
            target_pos=target_pos,
            select_unit_id=attacker.unit_id,
        )
    )


# ---------------------------------------------------------------------------
# 1. Base case: Sasha SCOP active, TANK does 9 HP to INFANTRY → +450g real-time
#    Phase 11J-L1-WAVE-2-SHIP: own SCOP-turn attacks credit immediately.
# ---------------------------------------------------------------------------


def test_war_bonds_base_9hp_to_infantry_credits_450_realtime() -> None:
    state = _make_state(p0_co_id=SASHA_CO_ID)
    state.co_states[0].war_bonds_active = True
    state.active_player = 0  # Sasha is the active player → real-time credit

    tank = _spawn(state, UnitType.TANK, 0, (0, 0), unit_id=1, hp=100)
    _spawn(state, UnitType.INFANTRY, 1, (0, 1), unit_id=2, hp=100)

    funds_before = state.funds[0]
    _attack(state, tank, (0, 1), override_dmg=90, override_counter=0)

    # Real-time — funds bumped immediately; pending stays at 0.
    assert state.funds[0] == funds_before + 9 * (1000 // 20)
    assert state.funds[0] == funds_before + 450
    assert state.co_states[0].pending_war_bonds_funds == 0


# ---------------------------------------------------------------------------
# 2. Cap: TANK kills MED_TANK (10 HP loss) → cap at 9 HP, +7200g (NOT 8000)
#    Real-time credit on activator's own SCOP turn.
# ---------------------------------------------------------------------------


def test_war_bonds_cap_at_9hp_on_kill_caps_payout() -> None:
    state = _make_state(p0_co_id=SASHA_CO_ID)
    state.co_states[0].war_bonds_active = True
    state.active_player = 0

    tank = _spawn(state, UnitType.TANK, 0, (0, 0), unit_id=1, hp=100)
    _spawn(state, UnitType.MED_TANK, 1, (0, 1), unit_id=2, hp=100)

    funds_before = state.funds[0]
    _attack(state, tank, (0, 1), override_dmg=100, override_counter=0)

    # min(10, 9) * 16000/20 = 9 * 800 = 7200, NOT 10 * 800 = 8000.
    # Credited real-time to funds on activator's own SCOP turn.
    assert state.funds[0] == funds_before + 9 * (16000 // 20)
    assert state.funds[0] == funds_before + 7200
    assert state.co_states[0].pending_war_bonds_funds == 0


# ---------------------------------------------------------------------------
# 3. SCOP inactive: same setup, no payout accrued
# ---------------------------------------------------------------------------


def test_war_bonds_scop_inactive_no_payout() -> None:
    state = _make_state(p0_co_id=SASHA_CO_ID)
    state.co_states[0].war_bonds_active = False  # explicit
    state.active_player = 0

    tank = _spawn(state, UnitType.TANK, 0, (0, 0), unit_id=1, hp=100)
    _spawn(state, UnitType.INFANTRY, 1, (0, 1), unit_id=2, hp=100)

    funds_before = state.funds[0]
    _attack(state, tank, (0, 1), override_dmg=90, override_counter=0)

    assert state.funds[0] == funds_before
    assert state.co_states[0].pending_war_bonds_funds == 0


# ---------------------------------------------------------------------------
# 4. Non-Sasha CO (Andy): no payout even with war_bonds_active erroneously set
# ---------------------------------------------------------------------------


def test_war_bonds_non_sasha_co_never_pays() -> None:
    state = _make_state(p0_co_id=ANDY_CO_ID)
    # Defensive: even if war_bonds_active is somehow set on a non-Sasha CO
    # state (it shouldn't be — only Sasha SCOP sets it), the payout helper
    # gates strictly on co_id == 19.
    state.co_states[0].war_bonds_active = True
    state.active_player = 0

    tank = _spawn(state, UnitType.TANK, 0, (0, 0), unit_id=1, hp=100)
    _spawn(state, UnitType.INFANTRY, 1, (0, 1), unit_id=2, hp=100)

    funds_before = state.funds[0]
    _attack(state, tank, (0, 1), override_dmg=90, override_counter=0)

    assert state.funds[0] == funds_before
    assert state.co_states[0].pending_war_bonds_funds == 0


# ---------------------------------------------------------------------------
# 5. Counter-attack payout: enemy attacks Sasha's defending TANK; Sasha's TANK
#    counter-deals 4 HP. Sasha (defender) earns 4 * cost(attacker) / 20.
# ---------------------------------------------------------------------------


def test_war_bonds_counter_attack_credits_defender_pending() -> None:
    """Sasha = P0. Enemy (Andy = P1) attacks Sasha's TANK; Sasha's TANK
    counter-deals 4 HP. Counter target = Andy's TANK (cost 7000)."""
    state = _make_state(p0_co_id=SASHA_CO_ID, p1_co_id=ANDY_CO_ID)
    state.co_states[0].war_bonds_active = True
    state.active_player = 1  # Andy is acting (his turn)

    enemy = _spawn(state, UnitType.TANK, 1, (0, 0), unit_id=10, hp=100)
    sasha_tank = _spawn(state, UnitType.TANK, 0, (0, 1), unit_id=11, hp=100)

    funds_before = state.funds[0]
    # Andy hits Sasha's TANK for 30 HP; Sasha counters with 40 HP (4 disp).
    _attack(state, enemy, (0, 1), override_dmg=30, override_counter=40)

    assert state.funds[0] == funds_before  # deferred
    # 4 disp damage * (TANK cost 7000 // 20) = 4 * 350 = 1400.
    assert state.co_states[0].pending_war_bonds_funds == 4 * (7000 // 20)
    assert state.co_states[0].pending_war_bonds_funds == 1400
    assert sasha_tank.hp == 70
    assert enemy.hp == 60


# ---------------------------------------------------------------------------
# 6. End-of-opponent-turn settlement: pending → funds, war_bonds_active cleared
# ---------------------------------------------------------------------------


def test_war_bonds_settlement_credits_funds_at_opp_end_turn() -> None:
    """Sasha activates SCOP, deals damage, accrues pending. Then Sasha ends
    turn (war_bonds_active stays True; pending stays). Then opponent ends
    turn — at this point pending is credited to Sasha's funds and the flag
    is cleared.
    """
    state = _make_state(p0_co_id=SASHA_CO_ID, p1_co_id=ANDY_CO_ID)
    state.co_states[0].war_bonds_active = True
    state.co_states[0].pending_war_bonds_funds = 1234
    state.funds[0] = 5000
    state.active_player = 0  # Sasha is acting

    state._end_turn()
    # After Sasha's own _end_turn: pending unchanged, flag still True.
    assert state.co_states[0].war_bonds_active is True
    assert state.co_states[0].pending_war_bonds_funds == 1234
    assert state.funds[0] == 5000
    assert state.active_player == 1

    state._end_turn()
    # After opp's _end_turn: pending credited and flag cleared.
    assert state.co_states[0].war_bonds_active is False
    assert state.co_states[0].pending_war_bonds_funds == 0
    assert state.funds[0] == 5000 + 1234


# ---------------------------------------------------------------------------
# 7. Settlement also fires when pending == 0 (pure flag-clear path)
# ---------------------------------------------------------------------------


def test_war_bonds_settlement_clears_flag_with_zero_pending() -> None:
    state = _make_state(p0_co_id=SASHA_CO_ID, p1_co_id=ANDY_CO_ID)
    state.co_states[0].war_bonds_active = True
    state.co_states[0].pending_war_bonds_funds = 0
    state.funds[0] = 7777
    state.active_player = 1  # opp is about to end turn

    state._end_turn()

    assert state.co_states[0].war_bonds_active is False
    assert state.funds[0] == 7777


# ---------------------------------------------------------------------------
# 8. SCOP activation primes war_bonds_active and resets stale pending
# ---------------------------------------------------------------------------


def test_war_bonds_scop_activation_primes_state() -> None:
    state = _make_state(p0_co_id=SASHA_CO_ID, p1_co_id=ANDY_CO_ID)
    # Stale pending from a hypothetical prior cycle should be cleared on
    # fresh activation so payouts don't double-credit.
    state.co_states[0].pending_war_bonds_funds = 999
    state.co_states[0].war_bonds_active = False
    state.active_player = 0

    state._apply_power_effects(player=0, cop=False)

    assert state.co_states[0].war_bonds_active is True
    assert state.co_states[0].pending_war_bonds_funds == 0
