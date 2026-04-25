"""Phase 11J-FINAL-STURM-SCOP-SHIP — Sturm "Meteor Strike" QA suite.

Pins AWBW canon for the Sturm COP/SCOP missile mechanic the engine ships
in Phase 11J-FINAL-STURM-SCOP-SHIP (lift of the Sturm SCOP freeze
imposed in Phase 11J-FINAL-BUILD-NO-OP-RESIDUALS).

Primary citations (every assertion below should match one of these):

* AWBW CO Chart (amarriner.com), Sturm row:
  https://awbw.amarriner.com/co.php
  *"Meteor Strike -- A 2-range missile deals 4 HP damage. The missile
    targets an enemy unit located at the greatest accumulation of unit
    value."*
  *"Meteor Strike II -- A 2-range missile deals 8 HP damage. The missile
    targets an enemy unit located at the greatest accumulation of unit
    value."*
* AWBW Wiki — Sturm: https://awbw.fandom.com/wiki/Sturm
* Wars Wiki — Sturm (Dual Strike anchor for the 0.1-display-HP floor
  shared by all flat-damage CO missiles):
  https://warswiki.org/wiki/Sturm

Empirical AOE shape verification (Phase 11J-FINAL recon):
  ``tools/_phase11j_sturm_aoe_verify.py`` replayed gid 1635679 (env 28
  SCOP @ missileCoords (9, 6); env 40 SCOP @ (8, 15)) and gid 1637200
  (env 12 COP @ (4, 7) — multi-strike). For every affected unit the
  Manhattan distance to the missileCoords centre was ≤ 2; the displayed
  HP delta was uniformly 8 (SCOP) / 4 (COP); units already at 1
  internal HP survived at 1 internal HP. 13-tile diamond confirmed.

Engine modelling decisions documented in
``docs/oracle_exception_audit/phase11j_final_sturm_1635679.md``:

* Damage: flat 40 internal HP (4 display) for COP, 80 internal HP
  (8 display) for SCOP, floored at 1 internal (~0.1 display). Same
  flooring rule the existing Hawke / Olaf / Von Bolt SCOP code uses.
* AOE: the engine ONLY applies damage when
  ``GameState._oracle_power_aoe_positions`` is set (oracle zip path:
  pinned from PHP ``missileCoords`` by ``tools/oracle_zip_replay.py``
  Sturm branch). When unset (RL / non-oracle path) the engine no-ops:
  no missile targeter is implemented and a global enemy -40/-80 would
  massively over-damage. One-shot: cleared after consumption.
* Targets: enemy units only (``self.units[opponent]``); friendly
  units never take Meteor Strike damage.
* No D2D modifiers, no luck, no terrain, no CO defence — flat scalar.
"""
from __future__ import annotations

from typing import Optional

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS


PLAIN = 1
NEUTRAL_BASE = 35

STURM_CO_ID = 29
ANDY_CO_ID = 1


_NEXT_UID = [21000]


def _make_state(
    *,
    width: int = 10,
    height: int = 10,
    terrain: Optional[list[list[int]]] = None,
    properties: Optional[list[PropertyState]] = None,
    p0_co: int = STURM_CO_ID,
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
        map_id=999_998,
        name="sturm_probe",
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


def _aoe_diamond(center: tuple[int, int]) -> set[tuple[int, int]]:
    """13-tile Manhattan diamond — matches ``tools/oracle_zip_replay.py``
    Sturm branch and the empirical AOE confirmed by
    ``tools/_phase11j_sturm_aoe_verify.py``."""
    cy, cx = center
    return {
        (cy + dr, cx + dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if abs(dr) + abs(dc) <= 2
    }


def _fire_meteor_strike(
    state: GameState, *, center: tuple[int, int], scop: bool
) -> None:
    """Fire Sturm's COP / SCOP with the oracle AOE pinned to the 2-range
    diamond. Charges the matching power bar so the activation step is
    legal under STEP-GATE."""
    state._oracle_power_aoe_positions = _aoe_diamond(center)
    co = state.co_states[state.active_player]
    if scop:
        co.power_bar = co._scop_threshold
        state.step(Action(ActionType.ACTIVATE_SCOP))
    else:
        co.power_bar = co._cop_threshold
        state.step(Action(ActionType.ACTIVATE_COP))


# ---------------------------------------------------------------------------
# 1. Damage values
# ---------------------------------------------------------------------------

def test_cop_deals_4hp_to_enemies_in_diamond():
    """AWBW CO Chart Sturm: Meteor Strike (COP) deals 4 HP. Engine
    internal scale: 4 display = 40 internal."""
    state = _make_state()
    enemies = []
    for pos in _aoe_diamond((5, 5)):
        enemies.append(_spawn(state, UnitType.TANK, player=1, pos=pos, hp=100))
    assert len(enemies) == 13
    _fire_meteor_strike(state, center=(5, 5), scop=False)
    for u in enemies:
        assert u.hp == 60, f"unit at {u.pos} expected hp=60, got {u.hp}"


def test_scop_deals_8hp_to_enemies_in_diamond():
    """AWBW CO Chart Sturm: Meteor Strike II (SCOP) deals 8 HP. Engine
    internal scale: 8 display = 80 internal. Empirically confirmed
    against gid 1635679 env 28 (7 affected units, all -8 display HP)."""
    state = _make_state()
    enemies = []
    for pos in _aoe_diamond((5, 5)):
        enemies.append(_spawn(state, UnitType.TANK, player=1, pos=pos, hp=100))
    assert len(enemies) == 13
    _fire_meteor_strike(state, center=(5, 5), scop=True)
    for u in enemies:
        assert u.hp == 20, f"unit at {u.pos} expected hp=20, got {u.hp}"


# ---------------------------------------------------------------------------
# 2. Damage floor — survives at 1 internal HP
# ---------------------------------------------------------------------------

def test_cop_floors_at_1_internal_hp():
    """A 1-display-HP unit (10 internal) hit by COP (40 nominal) survives
    at 1 internal HP (~0.1 display). Same flooring rule as Hawke / Olaf /
    Von Bolt SCOPs."""
    state = _make_state()
    u = _spawn(state, UnitType.INFANTRY, player=1, pos=(5, 5), hp=10)
    _fire_meteor_strike(state, center=(5, 5), scop=False)
    assert u.is_alive
    assert u.hp == 1


def test_scop_floors_at_1_internal_hp():
    """A 1-display-HP unit (10 internal) hit by SCOP (80 nominal) survives
    at 1 internal HP (~0.1 display)."""
    state = _make_state()
    u = _spawn(state, UnitType.INFANTRY, player=1, pos=(5, 5), hp=10)
    _fire_meteor_strike(state, center=(5, 5), scop=True)
    assert u.is_alive
    assert u.hp == 1


# ---------------------------------------------------------------------------
# 3. AOE shape — boundary (Manhattan ≤ 2)
# ---------------------------------------------------------------------------

def test_unit_just_outside_diamond_unaffected():
    """Pinned-AOE engine path: units outside the oracle-pinned diamond
    take no damage. (5, 8) is Manhattan 3 from (5, 5)."""
    state = _make_state()
    inside = _spawn(state, UnitType.TANK, player=1, pos=(5, 6), hp=100)
    outside = _spawn(state, UnitType.TANK, player=1, pos=(5, 8), hp=100)
    _fire_meteor_strike(state, center=(5, 5), scop=True)
    assert inside.hp == 20
    assert outside.hp == 100


def test_diagonal_tile_at_distance_2_is_inside():
    """The diamond IS Manhattan, not Chebyshev: (5+1, 5+1) is M=2 and
    inside; (5+2, 5+2) is M=4 and outside."""
    state = _make_state()
    inside_diag = _spawn(state, UnitType.TANK, player=1, pos=(6, 6), hp=100)
    outside_diag = _spawn(state, UnitType.TANK, player=1, pos=(7, 7), hp=100)
    _fire_meteor_strike(state, center=(5, 5), scop=False)
    assert inside_diag.hp == 60
    assert outside_diag.hp == 100


# ---------------------------------------------------------------------------
# 4. Friendly fire — own units NOT damaged
# ---------------------------------------------------------------------------

def test_friendly_units_not_damaged_cop():
    """Sturm's missile targets enemy units only. Friendly units inside
    the AOE take no damage on COP."""
    state = _make_state()
    own = _spawn(state, UnitType.TANK, player=0, pos=(5, 5), hp=100)
    own_corner = _spawn(state, UnitType.INFANTRY, player=0, pos=(5, 6), hp=80)
    _fire_meteor_strike(state, center=(5, 5), scop=False)
    assert own.hp == 100
    assert own_corner.hp == 80


def test_friendly_units_not_damaged_scop():
    """Sturm's missile targets enemy units only. Friendly units inside
    the AOE take no damage on SCOP."""
    state = _make_state()
    own = _spawn(state, UnitType.TANK, player=0, pos=(5, 5), hp=100)
    own_corner = _spawn(state, UnitType.INFANTRY, player=0, pos=(5, 6), hp=80)
    _fire_meteor_strike(state, center=(5, 5), scop=True)
    assert own.hp == 100
    assert own_corner.hp == 80


# ---------------------------------------------------------------------------
# 5. Oracle-pin one-shot — cleared after consumption
# ---------------------------------------------------------------------------

def test_aoe_pin_cleared_after_activation():
    """Defensive contract: ``_oracle_power_aoe_positions`` is consumed
    during the activation step. A subsequent activation with no fresh
    pin must NOT re-apply damage to the previous diamond."""
    state = _make_state()
    u = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    _fire_meteor_strike(state, center=(5, 5), scop=False)
    assert u.hp == 60
    assert state._oracle_power_aoe_positions is None


# ---------------------------------------------------------------------------
# 6. Non-oracle (RL) path — no AOE, no damage
# ---------------------------------------------------------------------------

def test_no_aoe_pin_means_no_damage():
    """RL / non-oracle path: no missile targeter is implemented for
    Sturm. With ``_oracle_power_aoe_positions = None`` the engine MUST
    no-op rather than nuke every enemy unit (which would break legality
    parity hard against AWBW). Documented in
    ``docs/oracle_exception_audit/phase11j_final_sturm_1635679.md``."""
    state = _make_state()
    u = _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    state._oracle_power_aoe_positions = None
    co = state.co_states[state.active_player]
    co.power_bar = co._scop_threshold
    state.step(Action(ActionType.ACTIVATE_SCOP))
    assert u.hp == 100
