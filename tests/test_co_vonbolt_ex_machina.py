"""Phase 11J-VONBOLT-SCOP-SHIP — Von Bolt "Ex Machina" QA suite.

These tests pin AWBW canon for the Von Bolt SCOP and the cross-cutting stun
mechanic the engine ships in Phase 11J-VONBOLT-SCOP-SHIP.

Primary citations (every assertion below should match one of these):

* AWBW Wiki — Von Bolt: https://awbw.fandom.com/wiki/Von_Bolt
  *"Ex Machina — A 2-range missile deals 3HP damage and prevents all
  affected enemy units from acting next turn. The missile targets the
  opponents' greatest accumulation of unit value."*
* AWBW CO Chart (amarriner.com): https://awbw.amarriner.com/co.php (Von Bolt
  row): *"A 2-range missile deals 3 HP damage and prevents all affected
  units from acting next turn. The missile targets the opponents' greatest
  accumulation of unit value."*
* Wars World News (Dual Strike — historical mechanic, supplementary): "A
  unit that has the stun status will enter wait mode on the army's next
  turn."  https://www.warsworldnews.com/wp/aw3/co-aw3/von-bolt-aw3/

Engine modeling decisions documented in
``docs/oracle_exception_audit/phase11j_vonbolt_scop_ship.md``:

* Damage: flat 30 internal HP / 3 display HP, floored at 1 internal (~0.1
  display). No luck, no terrain, no CO defense modifier — modeled in
  ``GameState._apply_power_effects`` co_id 30 SCOP branch alongside the
  other flat-loss SCOPs (Hawke, Olaf SCOP).
* Stun targets: enemy units only. The wiki is the more specific source
  ("affected *enemy* units"); the chart shorthand says "all affected
  units" but the wiki + cluster-B drift evidence both side with
  enemy-only.
* AOE: when ``GameState._oracle_power_aoe_positions`` is set (oracle
  zip path: pinned from PHP ``missileCoords``), only enemy units in that
  set are damaged + stunned. When unset (RL / non-oracle path), all
  enemy units are damaged + stunned (no missile targeter implemented;
  global stun is the safest legality posture). The oracle pin is the
  AWBW-canon 13-tile Manhattan diamond (2-range); see
  ``phase11j_missile_aoe_canon_sweep.md``.
* Stun lifetime: cleared in ``GameState._end_turn`` on the units of the
  player whose turn just ended. Ex Machina fires on Player A's turn →
  stun set on Player B's units → Player B's next turn is the served
  turn → cleared at the end of that served turn.
* Counter-attack: stunned defenders skip their counter (the act of
  counter-firing is itself an act and is forbidden by canon).
"""
from __future__ import annotations

from typing import Optional

import pytest

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    get_legal_actions,
)
from engine.co import make_co_state_safe
from engine.game import GameState, IllegalActionError
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS


# ---------------------------------------------------------------------------
# Terrain constants (mirrored from tests/test_engine_negative_legality.py)
# ---------------------------------------------------------------------------
PLAIN = 1
NEUTRAL_BASE = 35
OS_BASE = 39
BM_BASE = 44

VON_BOLT_CO_ID = 30
ANDY_CO_ID = 1


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
_NEXT_UID = [9000]


def _make_state(
    *,
    width: int = 10,
    height: int = 10,
    terrain: Optional[list[list[int]]] = None,
    properties: Optional[list[PropertyState]] = None,
    p0_co: int = VON_BOLT_CO_ID,
    p1_co: int = ANDY_CO_ID,
    active_player: int = 0,
    action_stage: ActionStage = ActionStage.SELECT,
) -> GameState:
    if terrain is None:
        terrain = [[PLAIN] * width for _ in range(height)]
    else:
        height = len(terrain)
        width = len(terrain[0])
    if properties is None:
        properties = []
    md = MapData(
        map_id=999_999,
        name="vonbolt_probe",
        map_type="std",
        terrain=[row[:] for row in terrain],
        height=height,
        width=width,
        cap_limit=999,
        unit_limit=999,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=properties,
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )
    state = GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(p0_co), make_co_state_safe(p1_co)],
        properties=properties,
        turn=1,
        active_player=active_player,
        action_stage=action_stage,
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
    return state


def _spawn(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    hp: int = 100,
    moved: bool = False,
) -> Unit:
    stats = UNIT_STATS[ut]
    _NEXT_UID[0] += 1
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
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


def _fire_ex_machina(state: GameState, *, center: tuple[int, int]) -> None:
    """Fire Von Bolt's SCOP with the oracle AOE pinned to the 2-range diamond.

    Matches ``tools/oracle_zip_replay.py`` Von Bolt branch: Manhattan distance
    ≤ 2 from ``center`` (13 tiles). Citations: AWBW CO Chart Von Bolt row
    (``https://awbw.amarriner.com/co.php``); Interface Guide 2-range diamond;
    PHP gid 1622328 env 28 ``unitReplace``.
    """
    cy, cx = center
    aoe = {
        (cy + dr, cx + dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if abs(dr) + abs(dc) <= 2
    }
    state._oracle_power_aoe_positions = aoe
    # Charge the bar so ACTIVATE_SCOP is legal under STEP-GATE.
    co = state.co_states[state.active_player]
    co.power_bar = co._scop_threshold
    state.step(Action(ActionType.ACTIVATE_SCOP))


# ---------------------------------------------------------------------------
# 1. Damage tests
# ---------------------------------------------------------------------------

def test_damage_3hp_to_all_enemies_in_2_range_diamond():
    """AWBW Wiki Von Bolt: Ex Machina deals 3 HP damage to all affected enemy
    units. Engine internal scale: 3 display HP = 30 internal HP. One enemy per
    tile of the 13-tile Manhattan diamond (2-range) around (5, 5).
    """
    state = _make_state()
    enemies = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) + abs(dc) > 2:
                continue
            u = _spawn(
                state, UnitType.TANK, player=1, pos=(5 + dr, 5 + dc), hp=100
            )
            enemies.append(u)
    assert len(enemies) == 13
    _fire_ex_machina(state, center=(5, 5))
    for u in enemies:
        assert u.hp == 70, f"unit at {u.pos} expected hp=70, got {u.hp}"


def test_damage_floored_at_1_internal_hp():
    """Wars Wiki Dual Strike anchor (mirrored on AWBW Fandom): "leaving at
    least 0.1 HP." A unit at 10 internal HP (1 display HP) hit by Ex Machina
    survives at 1 internal HP (~0.1 display) — does NOT die from the 30-HP
    nominal damage. Same flooring rule the existing Hawke/Olaf SCOP code uses.
    """
    state = _make_state()
    u = _spawn(state, UnitType.INFANTRY, player=1, pos=(5, 5), hp=10)
    _fire_ex_machina(state, center=(5, 5))
    assert u.is_alive
    assert u.hp == 1


# ---------------------------------------------------------------------------
# 2. Stun-flag set test
# ---------------------------------------------------------------------------

def test_stun_flag_set_on_enemies_in_aoe():
    """AWBW Wiki Von Bolt: stun "prevents all affected enemy units from
    acting next turn." Engine sets ``Unit.is_stunned`` on every enemy in the
    AOE; own units are NOT stunned (wiki specifies "enemy units")."""
    state = _make_state()
    own_inside = _spawn(state, UnitType.TANK, player=0, pos=(5, 5), hp=100)
    enemy_inside = _spawn(state, UnitType.TANK, player=1, pos=(5, 6), hp=100)
    enemy_outside = _spawn(state, UnitType.TANK, player=1, pos=(8, 8), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    assert enemy_inside.is_stunned
    assert not own_inside.is_stunned
    assert not enemy_outside.is_stunned


# ---------------------------------------------------------------------------
# 3-7. Stun blocks legal actions on next opponent turn
# ---------------------------------------------------------------------------

def _advance_to_opponent_turn(state: GameState) -> None:
    """End Von Bolt's turn after firing the SCOP so opponent becomes active."""
    # Drain any unmoved Von Bolt units to satisfy END_TURN gate, then end.
    for u in state.units[state.active_player]:
        u.moved = True
    state.step(Action(ActionType.END_TURN))


def test_stun_blocks_select_unit_in_legal_mask():
    """STEP-GATE invariant — Von Bolt fires Ex Machina; on the opponent's
    next turn, ``get_legal_actions`` MUST NOT contain SELECT_UNIT for any
    stunned unit. AWBW canon: Wiki Von Bolt — *"prevents all affected
    enemy units from acting next turn."*"""
    state = _make_state()
    stunned = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    free = _spawn(state, UnitType.TANK, player=1, pos=(8, 8), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)
    legal = get_legal_actions(state)
    select_actions = [a for a in legal if a.action_type == ActionType.SELECT_UNIT]
    select_positions = {a.unit_pos for a in select_actions}
    assert stunned.pos not in select_positions
    assert free.pos in select_positions


def test_stun_blocks_attack_via_step_gate():
    """Stunned unit cannot ATTACK on its owner's next turn — STEP-GATE
    rejects because the SELECT_UNIT was never offered. AWBW canon: Wiki
    Von Bolt — stun blocks all next-turn actions."""
    state = _make_state()
    stunned = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    _spawn(state, UnitType.INFANTRY, player=0, pos=(5, 6), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)
    select = Action(ActionType.SELECT_UNIT, unit_pos=stunned.pos)
    with pytest.raises(IllegalActionError):
        state.step(select)


def test_stun_blocks_capture_via_step_gate():
    """Stunned Infantry on an enemy property cannot CAPTURE on its owner's
    next turn. AWBW canon: Wiki Von Bolt — stun blocks all next-turn acts."""
    prop = PropertyState(
        terrain_id=NEUTRAL_BASE, row=5, col=5, owner=None, capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False,
        is_base=True, is_airport=False, is_port=False,
    )
    terrain = [[PLAIN] * 10 for _ in range(10)]
    terrain[5][5] = NEUTRAL_BASE
    state = _make_state(terrain=terrain, properties=[prop])
    stunned = _spawn(state, UnitType.INFANTRY, player=1, pos=(5, 5), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)
    select = Action(ActionType.SELECT_UNIT, unit_pos=stunned.pos)
    with pytest.raises(IllegalActionError):
        state.step(select)


def test_stun_blocks_wait_via_step_gate():
    """Even WAIT (no-op) is forbidden — the stunned unit cannot be the
    subject of any action on its owner's next turn. AWBW canon: Wiki Von
    Bolt — *"prevents all affected enemy units from acting."* The act of
    WAIT is still an act (it commits the unit). STEP-GATE rejects."""
    state = _make_state()
    stunned = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)
    wait = Action(ActionType.WAIT, unit_pos=stunned.pos, move_pos=stunned.pos)
    with pytest.raises(IllegalActionError):
        state.step(wait)


# ---------------------------------------------------------------------------
# 8. Stun clears at end of served opponent turn
# ---------------------------------------------------------------------------

def test_stun_8a_blocks_during_opponents_turn():
    """Imperator-confirmed timing pin (Phase 11J-VONBOLT-SCOP-SHIP, stun
    timing correction): stun is set on Von Bolt's turn T, BLOCKS during
    opponent's turn T+1. At the very start of T+1, ``is_stunned == True``
    and the unit is not in ``get_legal_actions``. STEP-GATE rejects any
    direct attempt to act."""
    state = _make_state()
    stunned = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)
    assert state.active_player == 1, "should be opponent's turn T+1"
    assert stunned.is_stunned, "stun must be active at START of T+1"
    legal = get_legal_actions(state)
    select_positions = {
        a.unit_pos for a in legal if a.action_type == ActionType.SELECT_UNIT
    }
    assert stunned.pos not in select_positions, "stunned unit must be filtered from mask"
    with pytest.raises(IllegalActionError):
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=stunned.pos))


def test_stun_8b_clears_at_end_of_opponents_served_turn():
    """Imperator-confirmed timing pin: clear fires at the END of T+1, NOT
    the start. After the opponent's ``_end_turn`` runs, ``is_stunned`` must
    be ``False``. The clear is on the player whose turn just ended (the
    served army), so a 2-player round resolves: P0(T)→P1(T+1)→clear→P0(T+2)→P1(T+3 free)."""
    state = _make_state()
    stunned = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)
    assert stunned.is_stunned
    state.step(Action(ActionType.END_TURN))
    assert not stunned.is_stunned, "stun must clear at END of opponent's served turn"


def test_stun_8c_does_not_clear_at_start_of_opponents_turn():
    """Imperator-confirmed timing pin (negative): the clear is bound to the
    END of the served turn, NOT the START. If the implementation moves the
    clear to a start-of-turn hook, the stun has zero effect — the opponent
    would simply move the unit on T+1 unimpeded. This test would fail loud
    if that regression slipped in."""
    state = _make_state()
    stunned = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)
    # We are now AT START of T+1 (opponent active, no actions yet taken).
    assert state.active_player == 1
    assert stunned.is_stunned, (
        "regression guard: stun MUST NOT have been cleared by a "
        "start-of-turn hook on T+1"
    )


def test_stun_8d_lasts_exactly_one_opponent_turn():
    """Imperator-confirmed timing pin: stun lasts exactly one served opponent
    turn. By the start of the opponent's NEXT-NEXT turn (T+3 in the original
    schedule), the previously-stunned unit is free and re-enters the legal
    SELECT_UNIT mask. Walks the full T → T+1 → T+2 → T+3 cycle."""
    state = _make_state()
    stunned = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    # End T (Von Bolt). Now T+1 (opponent, served turn).
    _advance_to_opponent_turn(state)
    assert state.active_player == 1
    assert stunned.is_stunned
    # End T+1 (opponent's served turn). Clear fires here. Now T+2 (Von Bolt).
    state.step(Action(ActionType.END_TURN))
    assert state.active_player == 0
    assert not stunned.is_stunned, "must be cleared by end of T+1"
    # End T+2 (Von Bolt has nothing — we never spawned P0 units). Now T+3.
    state.step(Action(ActionType.END_TURN))
    assert state.active_player == 1, "should be opponent's free turn T+3"
    assert not stunned.is_stunned
    legal = get_legal_actions(state)
    select_positions = {
        a.unit_pos for a in legal if a.action_type == ActionType.SELECT_UNIT
    }
    assert stunned.pos in select_positions, (
        "previously-stunned unit MUST be in legal mask on T+3 (the next "
        "own turn after the served one)"
    )


# ---------------------------------------------------------------------------
# 9. Stun blocks counter-attack
# ---------------------------------------------------------------------------

def test_stun_blocks_counter_attack():
    """Wiki Von Bolt: stun blocks ALL acts on the next turn. A counter-attack
    is an act; PHP correctly skips it on a stunned defender. The pre-fix
    engine ran the counter regardless, taking attacker HP with it — the
    cluster-B drift mechanism in gids 1621434 / 1621898 / 1622328 (see
    ``docs/oracle_exception_audit/phase11j_vonbolt_scop_ship.md`` §3)."""
    state = _make_state()
    # Von Bolt fires Ex Machina centered on the defender, then attacks.
    stunned_defender = _spawn(
        state, UnitType.TANK, player=1, pos=(5, 5), hp=100
    )
    attacker = _spawn(state, UnitType.TANK, player=0, pos=(5, 6), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    assert stunned_defender.is_stunned
    pre_atk_hp = attacker.hp
    pre_def_hp = stunned_defender.hp  # 70 after Ex Machina
    # Apply attack via the engine's internal path (bypass STEP-GATE because
    # the SCOP envelope is the activator's turn — Von Bolt can still attack).
    state.action_stage = ActionStage.ACTION
    state.selected_unit = attacker
    state.selected_move_pos = attacker.pos
    state.step(
        Action(
            ActionType.ATTACK,
            unit_pos=attacker.pos,
            move_pos=attacker.pos,
            target_pos=stunned_defender.pos,
        ),
        oracle_mode=True,
    )
    assert attacker.hp == pre_atk_hp, (
        f"stunned defender must skip counter; attacker hp dropped "
        f"{pre_atk_hp} -> {attacker.hp}"
    )
    assert stunned_defender.hp < pre_def_hp, "primary attack still applies"


def test_unstunned_defender_still_counters():
    """Negative control for the stun-blocks-counter test: a non-stunned
    defender DOES counter (regression guard against accidentally disabling
    counter-fire altogether)."""
    state = _make_state(p0_co=ANDY_CO_ID, p1_co=ANDY_CO_ID)
    defender = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    attacker = _spawn(state, UnitType.TANK, player=0, pos=(5, 6), hp=100)
    pre_atk = attacker.hp
    state.action_stage = ActionStage.ACTION
    state.selected_unit = attacker
    state.selected_move_pos = attacker.pos
    state.step(
        Action(
            ActionType.ATTACK,
            unit_pos=attacker.pos,
            move_pos=attacker.pos,
            target_pos=defender.pos,
        ),
        oracle_mode=True,
    )
    assert attacker.hp < pre_atk, "non-stunned defender must counter"


# ---------------------------------------------------------------------------
# 10. AOE boundary
# ---------------------------------------------------------------------------

def test_unit_just_outside_diamond_unaffected():
    """Pinned-AOE engine path: units outside the oracle-pinned diamond
    take no damage and no stun. (5, 8) is Manhattan 3 from (5, 5)."""
    state = _make_state()
    inside = _spawn(state, UnitType.TANK, player=1, pos=(5, 6), hp=100)
    outside = _spawn(state, UnitType.TANK, player=1, pos=(5, 8), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    assert inside.hp == 70
    assert inside.is_stunned
    assert outside.hp == 100
    assert not outside.is_stunned


# ---------------------------------------------------------------------------
# 11. Friendly fire — own units NOT damaged or stunned
# ---------------------------------------------------------------------------

def test_friendly_units_not_damaged_or_stunned():
    """Wiki Von Bolt is explicit: damage + stun apply to "all affected
    enemy units." The engine's flat-loss SCOP code path iterates over
    ``self.units[opponent]`` only — friendlies are skipped entirely."""
    state = _make_state()
    own = _spawn(state, UnitType.TANK, player=0, pos=(5, 5), hp=100)
    own_corner = _spawn(state, UnitType.INFANTRY, player=0, pos=(5, 6), hp=80)
    _fire_ex_machina(state, center=(5, 5))
    assert own.hp == 100
    assert own_corner.hp == 80
    assert not own.is_stunned
    assert not own_corner.is_stunned


# ---------------------------------------------------------------------------
# 12. Property-style invariant: every SELECT in get_legal_actions succeeds
#     under STEP-GATE; every stunned unit is rejected.
# ---------------------------------------------------------------------------

def test_property_legal_mask_consistent_with_step_gate():
    """Property invariant: for every SELECT_UNIT in get_legal_actions,
    state.step accepts it (no STEP-GATE rejection); for every stunned
    unit, state.step rejects it. Together these prove the legality-mask
    closure for stun (Phase 11J-VONBOLT-SCOP-SHIP)."""
    state = _make_state()
    stunned = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    free_a = _spawn(state, UnitType.TANK, player=1, pos=(8, 8), hp=100)
    free_b = _spawn(state, UnitType.INFANTRY, player=1, pos=(2, 2), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)

    legal = get_legal_actions(state)
    select_actions = [a for a in legal if a.action_type == ActionType.SELECT_UNIT]
    select_positions = {a.unit_pos for a in select_actions}

    # Every offered SELECT is acceptable.
    for a in select_actions:
        snapshot_stage = state.action_stage
        snapshot_sel = state.selected_unit
        try:
            state.step(a)
        finally:
            state.action_stage = snapshot_stage
            state.selected_unit = snapshot_sel

    # Stunned unit is never offered AND step rejects direct attempt.
    assert stunned.pos not in select_positions
    with pytest.raises(IllegalActionError):
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=stunned.pos))

    # Free units ARE offered.
    assert free_a.pos in select_positions
    assert free_b.pos in select_positions


# ---------------------------------------------------------------------------
# 13. END_TURN remains legal even when only stunned units exist.
# ---------------------------------------------------------------------------

def test_end_turn_legal_when_only_stunned_units_remain():
    """If an army is fully stunned, END_TURN must remain legal — otherwise
    the engine deadlocks on a turn the AWBW player can pass freely. This
    matches AWBW behavior: the player simply ends their turn."""
    state = _make_state()
    s1 = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    s2 = _spawn(state, UnitType.INFANTRY, player=1, pos=(5, 6), hp=100)
    _fire_ex_machina(state, center=(5, 5))
    _advance_to_opponent_turn(state)
    legal = get_legal_actions(state)
    end_turn_actions = [a for a in legal if a.action_type == ActionType.END_TURN]
    assert end_turn_actions, "END_TURN must be legal when all units are stunned"
    # And it must execute cleanly.
    state.step(Action(ActionType.END_TURN))
    assert not s1.is_stunned
    assert not s2.is_stunned
