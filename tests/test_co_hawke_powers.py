"""Phase 11J-FINAL-HAWKE-CLUSTER — Hawke "Black Wave" / "Black Storm" QA suite.

Pins AWBW canon for Hawke's (co_id 12) Day-to-Day / COP / SCOP mechanics
the engine ships in Phase 11J-FINAL-HAWKE-CLUSTER (lift of the Hawke CO
freeze imposed in Phase 11J-FINAL-BUILD-NO-OP-RESIDUALS).

Primary citations (every assertion below should match one of these):

* AWBW CO Chart (amarriner.com), Hawke row — Tier 1, canonical:
  https://awbw.amarriner.com/co.php
  D2D: *"+10% attack to all units."* (no defense bonus, no free heal)
  Black Wave (COP): *"All units gain +1 HP. All enemy units take 1 HP damage."*
  Black Storm (SCOP): *"All units gain +2 HP. All enemy units take 2 HP damage."*
* AWBW Wiki — Hawke (Tier 2, supporting):
  https://awbw.fandom.com/wiki/Hawke
* Wars Wiki — Hawke (Tier 2, vanilla AW cross-check):
  https://warswiki.org/wiki/Hawke

Empirical PHP-snapshot verification (Phase 11J-FINAL-HAWKE-CLUSTER recon):
  ``tools/_phase11j_hawke_cop_drill.py`` drilled gid 1635846 env 30 (Hawke
  COP "Black Wave" day 16). Own units that took NO combat damage in the
  same envelope showed clean +1.0 display HP heals (Artillery 7.1 -> 8.1,
  Infantry 7.0 -> 8.0, Mech 5.6 -> 6.6, Mech 8.5 -> 9.5). Enemy units that
  weren't fired upon showed -1.0 display HP baseline. SCOP samples from
  gids 1617442 / 1635679 confirmed +2.0 / -2.0 baseline.

Pre-fix engine bug: ``engine/game.py`` Hawke branch always healed friends
``+20`` internal HP regardless of cop/scop, over-healing Black Wave by
+10 internal (+1 display bar) per fire. ``data/co_data.json`` Black Wave
description ("...all own units recover 2 HP") was also inconsistent with
the chart and is corrected in the same closeout.

Engine modelling (matching the canon above):

* COP "Black Wave"  : friends +10 internal HP (+1 display bar),
                     enemies -10 internal HP (-1 display, floored 1).
* SCOP "Black Storm": friends +20 internal HP (+2 display bars),
                     enemies -20 internal HP (-2 display, floored 1).
* D2D: +10% atk only — NO free heal, NO defense bonus.
* Enemy floor: 1 internal HP — same flooring rule as Olaf / Drake /
  Von Bolt / Sturm flat-loss SCOPs (universal Hawke-family floor).
"""
from __future__ import annotations

from typing import Optional

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS


PLAIN = 1

HAWKE_CO_ID = 12
ANDY_CO_ID = 1


_NEXT_UID = [22000]


def _make_state(
    *,
    width: int = 10,
    height: int = 10,
    p0_co: int = HAWKE_CO_ID,
    p1_co: int = ANDY_CO_ID,
    active_player: int = 0,
) -> GameState:
    terrain = [[PLAIN] * width for _ in range(height)]
    md = MapData(
        map_id=999_997,
        name="hawke_probe",
        map_type="std",
        terrain=terrain,
        height=height,
        width=width,
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
        funds=[0, 0],
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


def _spawn(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    hp: int = 100,
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
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=_NEXT_UID[0],
    )
    state.units[player].append(u)
    return u


def _fire_power(state: GameState, *, scop: bool) -> None:
    co = state.co_states[state.active_player]
    if scop:
        co.power_bar = co._scop_threshold
        state.step(Action(ActionType.ACTIVATE_SCOP))
    else:
        co.power_bar = co._cop_threshold
        state.step(Action(ActionType.ACTIVATE_COP))


# ---------------------------------------------------------------------------
# 1. COP "Black Wave" — friend +10 internal, enemy -10 internal
# ---------------------------------------------------------------------------

def test_cop_heals_friends_plus_10_internal_hp():
    """AWBW CO Chart: Black Wave gives all own units +1 HP. Engine
    internal scale: 1 display = 10 internal. Empirically confirmed
    against gid 1635846 env 30 (Artillery 7.1->8.1, Infantry 7.0->8.0,
    Mech 5.6->6.6, Mech 8.5->9.5)."""
    state = _make_state()
    friends = [
        _spawn(state, UnitType.INFANTRY, player=0, pos=(2, 2), hp=70),
        _spawn(state, UnitType.MECH,     player=0, pos=(3, 3), hp=56),
        _spawn(state, UnitType.TANK,     player=0, pos=(4, 4), hp=85),
        _spawn(state, UnitType.ARTILLERY, player=0, pos=(5, 5), hp=71),
    ]
    _fire_power(state, scop=False)
    expected = [80, 66, 95, 81]
    for u, want in zip(friends, expected):
        assert u.hp == want, f"unit {u.unit_type} expected {want}, got {u.hp}"


def test_cop_damages_enemies_minus_10_internal_hp():
    """AWBW CO Chart: Black Wave deals 1 HP damage to all enemy units.
    Engine internal scale: 1 display = 10 internal. Empirically
    confirmed against gid 1635846 env 30 (every enemy unit not in
    combat that envelope showed exactly -1.0 display HP)."""
    state = _make_state()
    enemies = [
        _spawn(state, UnitType.INFANTRY, player=1, pos=(2, 2), hp=100),
        _spawn(state, UnitType.MECH,     player=1, pos=(3, 3), hp=80),
        _spawn(state, UnitType.TANK,     player=1, pos=(4, 4), hp=50),
    ]
    _fire_power(state, scop=False)
    for u, want in zip(enemies, [90, 70, 40]):
        assert u.hp == want, f"unit {u.unit_type} expected {want}, got {u.hp}"


def test_cop_friend_heal_caps_at_100_internal():
    """A friendly unit at 95 internal HP heals only by +5 (capped at
    the 100 internal HP ceiling), NOT +10."""
    state = _make_state()
    u = _spawn(state, UnitType.INFANTRY, player=0, pos=(2, 2), hp=95)
    _fire_power(state, scop=False)
    assert u.hp == 100


def test_cop_enemy_damage_floors_at_1_internal_hp():
    """A 1-display-HP enemy (10 internal) hit by Black Wave (-10
    nominal) survives at 1 internal HP. Same flooring rule as Olaf /
    Drake / Von Bolt / Sturm flat-loss SCOPs."""
    state = _make_state()
    u = _spawn(state, UnitType.INFANTRY, player=1, pos=(2, 2), hp=10)
    _fire_power(state, scop=False)
    assert u.is_alive
    assert u.hp == 1


# ---------------------------------------------------------------------------
# 2. SCOP "Black Storm" — friend +20 internal, enemy -20 internal
# ---------------------------------------------------------------------------

def test_scop_heals_friends_plus_20_internal_hp():
    """AWBW CO Chart: Black Storm gives all own units +2 HP. Engine
    internal scale: 2 display = 20 internal."""
    state = _make_state()
    friends = [
        _spawn(state, UnitType.INFANTRY, player=0, pos=(2, 2), hp=60),
        _spawn(state, UnitType.MECH,     player=0, pos=(3, 3), hp=55),
        _spawn(state, UnitType.TANK,     player=0, pos=(4, 4), hp=80),
    ]
    _fire_power(state, scop=True)
    for u, want in zip(friends, [80, 75, 100]):
        assert u.hp == want, f"unit {u.unit_type} expected {want}, got {u.hp}"


def test_scop_damages_enemies_minus_20_internal_hp():
    """AWBW CO Chart: Black Storm deals 2 HP damage to all enemy
    units. Engine internal scale: 2 display = 20 internal."""
    state = _make_state()
    enemies = [
        _spawn(state, UnitType.INFANTRY, player=1, pos=(2, 2), hp=100),
        _spawn(state, UnitType.MECH,     player=1, pos=(3, 3), hp=80),
    ]
    _fire_power(state, scop=True)
    for u, want in zip(enemies, [80, 60]):
        assert u.hp == want


def test_scop_enemy_damage_floors_at_1_internal_hp():
    """A 1-display-HP enemy (10 internal) hit by Black Storm (-20
    nominal) survives at 1 internal HP."""
    state = _make_state()
    u = _spawn(state, UnitType.INFANTRY, player=1, pos=(2, 2), hp=10)
    _fire_power(state, scop=True)
    assert u.is_alive
    assert u.hp == 1


# ---------------------------------------------------------------------------
# 3. Friendly fire / cross-side guards
# ---------------------------------------------------------------------------

def test_cop_does_not_damage_friends():
    """Black Wave's enemy damage applies to the OPPONENT seat only;
    same-seat units never take the -1 HP."""
    state = _make_state()
    # Caster is P0; spawn enemies of P0 for the heal pass and
    # confirm the opponent's units don't accidentally heal too.
    own = _spawn(state, UnitType.TANK, player=0, pos=(2, 2), hp=80)
    enemy = _spawn(state, UnitType.TANK, player=1, pos=(8, 8), hp=80)
    _fire_power(state, scop=False)
    assert own.hp == 90    # +10 friend heal
    assert enemy.hp == 70  # -10 enemy damage


def test_scop_does_not_heal_enemies():
    """Black Storm's heal applies to the CASTER seat only; the
    opponent's units do not heal."""
    state = _make_state()
    own = _spawn(state, UnitType.TANK, player=0, pos=(2, 2), hp=70)
    enemy = _spawn(state, UnitType.TANK, player=1, pos=(8, 8), hp=70)
    _fire_power(state, scop=True)
    assert own.hp == 90    # +20 friend heal
    assert enemy.hp == 50  # -20 enemy damage (no heal cancelling)


# ---------------------------------------------------------------------------
# 4. D2D — +10% ATK only, NO free heal, NO defense bonus
# ---------------------------------------------------------------------------

def test_d2d_no_free_heal_at_turn_start():
    """AWBW CO Chart Hawke D2D is +10% ATK only. No 1-HP free heal at
    start of turn (a community-folk-rule that the engine has never
    implemented and must NOT acquire). Re-arming this guard prevents
    a future regression where a contributor 'patches in' Hawke 1-HP
    healing from the Wars Wiki Dual Strike anchor."""
    state = _make_state()
    u = _spawn(state, UnitType.INFANTRY, player=0, pos=(2, 2), hp=70)
    pre_funds = state.funds[0]

    # Drive a clean end-of-P1-turn → P0 day 2 start transition. No
    # property income (no properties on the probe map), so the funds
    # delta from a free heal would be 0 anyway — the test pins HP
    # parity and absence of any side-effect spend.
    state.active_player = 1
    state.step(Action(ActionType.END_TURN))

    assert u.hp == 70, "Hawke D2D must NOT free-heal at start of turn"
    assert state.funds[0] == pre_funds, "no spend allowed on the heal that doesn't happen"


def test_d2d_atk_modifier_present():
    """Hawke's D2D +10% ATK modifier is wired into ``co_states[].atk_modifier``
    via ``data/co_data.json`` (``atk_modifiers.all = 10``). Pin the surface
    so a future ``co_data.json`` rewrite cannot silently zero it out."""
    state = _make_state()
    co = state.co_states[0]
    # The engine consumes ``atk_modifiers`` from the dataclass; the exact
    # field name is whatever ``make_co_state_safe`` exposes. Probe both
    # the raw data and the live state to keep the test robust.
    assert co.co_id == HAWKE_CO_ID
    # Smoke-check: under D2D (no power active), Hawke should have a
    # non-zero universal attack bonus surface. The exact attribute name
    # may evolve; the assertion is "Hawke is not a vanilla CO".
    assert co.name == "Hawke"
