"""Phase 11J-RACHEL-SCOP-COVERING-FIRE-SHIP — Rachel "Covering Fire" QA suite.

Pins AWBW canon for Rachel's SCOP and the new Counter-multiplicity contract on
``GameState._oracle_power_aoe_positions`` introduced for the 3-missile case.

Primary citations:

* AWBW CO Chart (amarriner.com): https://awbw.amarriner.com/co.php (Rachel
  row): *"Covering Fire — Three 2-range missiles deal 3 HP damage each. The
  missiles target the opponents' greatest accumulation of footsoldier HP,
  unit value, and unit HP (in that order)."*
* AWBW Fandom Wiki — Rachel: https://awbw.fandom.com/wiki/Rachel — same
  3-missile / 3 HP / 2-range mechanic.

Engine modeling decisions documented in
``docs/oracle_exception_audit/phase11j_rachel_scop_covering_fire_ship.md``:

* Damage per missile: flat 30 internal HP / 3 display HP, floored at 1
  internal — same flat-loss family as Hawke / Olaf SCOP / Von Bolt SCOP.
* Multiplicity: oracle pin is a ``Counter[(row, col)] -> hit_count``; a
  unit at a tile hit by N overlapping missiles takes ``30 * N`` HP. Drilled
  on gid 1622501 env 20 where two missiles share (y=11, x=20).
* Targeting: enemy units only. Chart text targets "the opponents'"
  accumulations; mirrors the Von Bolt enemy-only convention.
* AOE shape: oracle pins the AWBW-canon Manhattan diamond per missile (Von
  Bolt SCOP uses the same 13-tile family; see phase11j_missile_aoe_canon_sweep).
* No oracle pin: SCOP fires no-op (engine alone cannot decide where the
  missiles land — out-of-band targeter required).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import pytest

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS


PLAIN = 1
RACHEL_CO_ID = 28
ANDY_CO_ID = 1

_NEXT_UID = [12000]


def _make_state(
    *,
    width: int = 25,
    height: int = 25,
    p0_co: int = RACHEL_CO_ID,
    p1_co: int = ANDY_CO_ID,
) -> GameState:
    terrain = [[PLAIN] * width for _ in range(height)]
    md = MapData(
        map_id=999_998,
        name="rachel_probe",
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


def _spawn(state: GameState, ut: UnitType, player: int, pos: tuple[int, int], *, hp: int = 100) -> Unit:
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


def _build_aoe_counter(centers: list[tuple[int, int]]) -> Counter:
    """Mirror the oracle pin: expand each missile center to its 3x3 box and
    accumulate per-tile hit counts (overlapping missiles stack)."""
    c: Counter = Counter()
    for cy, cx in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                c[(cy + dr, cx + dc)] += 1
    return c


def _fire_covering_fire(state: GameState, *, centers: list[tuple[int, int]]) -> None:
    state._oracle_power_aoe_positions = _build_aoe_counter(centers)
    co = state.co_states[state.active_player]
    co.power_bar = co._scop_threshold
    state.step(Action(ActionType.ACTIVATE_SCOP))


# ---------------------------------------------------------------------------
# 1. Three missiles each dealing 30 HP to enemies in their 3x3 areas.
# ---------------------------------------------------------------------------

def test_three_missiles_each_apply_3hp_damage_in_3x3():
    """Chart canon: three independent 3 HP / 3x3 missiles. Spawn enemies in
    each missile's 3x3 (centers are non-overlapping). Each enemy takes
    exactly 30 internal HP (3 display HP)."""
    state = _make_state()
    centers = [(5, 5), (10, 10), (15, 15)]
    enemies = [
        _spawn(state, UnitType.TANK, player=1, pos=c, hp=100) for c in centers
    ]
    _fire_covering_fire(state, centers=centers)
    for u in enemies:
        assert u.hp == 70, f"enemy at {u.pos}: expected 70 HP, got {u.hp}"


# ---------------------------------------------------------------------------
# 2. Damage clamps at 1 internal HP (matching the flat-loss SCOP family).
# ---------------------------------------------------------------------------

def test_damage_floored_at_1_internal_hp():
    """Hawke / Olaf / Von Bolt parity: a low-HP enemy hit by Rachel's SCOP
    survives at 1 internal HP rather than dying. Rachel's missile is the
    same flat-loss family — no kills via SCOP damage."""
    state = _make_state()
    enemy = _spawn(state, UnitType.INFANTRY, player=1, pos=(5, 5), hp=10)
    _fire_covering_fire(state, centers=[(5, 5), (12, 12), (18, 18)])
    assert enemy.is_alive
    assert enemy.hp == 1


# ---------------------------------------------------------------------------
# 3. Friendly units NOT damaged (chart targets "the opponents'" units).
# ---------------------------------------------------------------------------

def test_friendly_units_inside_aoe_not_damaged():
    """Chart text: missiles "target the opponents' greatest accumulation."
    Rachel's own units inside the AOE take no damage — same enemy-only
    convention used by Von Bolt's SCOP."""
    state = _make_state()
    own_inside = _spawn(state, UnitType.TANK, player=0, pos=(5, 5), hp=100)
    enemy_inside = _spawn(state, UnitType.TANK, player=1, pos=(5, 6), hp=100)
    _fire_covering_fire(state, centers=[(5, 5), (12, 12), (18, 18)])
    assert own_inside.hp == 100
    assert enemy_inside.hp == 70


# ---------------------------------------------------------------------------
# 4. Empty AOE: pin valid positions but no units in AOE → no damage, no error.
# ---------------------------------------------------------------------------

def test_empty_aoe_no_damage_no_error():
    """Engine must tolerate a pin that lands on empty terrain (no enemies in
    any of the 3 missiles' 3x3 boxes). No exception, no damage."""
    state = _make_state()
    far_enemy = _spawn(state, UnitType.TANK, player=1, pos=(20, 20), hp=100)
    _fire_covering_fire(state, centers=[(2, 2), (5, 5), (8, 8)])
    assert far_enemy.hp == 100


# ---------------------------------------------------------------------------
# 5. Channel reset after consume — one-shot contract (mirror of Von Bolt).
# ---------------------------------------------------------------------------

def test_channel_reset_after_consume():
    """``_oracle_power_aoe_positions`` is one-shot — after Rachel SCOP fires,
    the channel reverts to ``None`` so a subsequent power activation does
    NOT re-apply the same AOE."""
    state = _make_state()
    _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100)
    _fire_covering_fire(state, centers=[(5, 5), (10, 10), (15, 15)])
    assert state._oracle_power_aoe_positions is None


# ---------------------------------------------------------------------------
# 6. No oracle pin → SCOP fires gracefully with no damage.
# ---------------------------------------------------------------------------

def test_no_oracle_pin_falls_back_to_no_damage():
    """RL / non-oracle path: the engine alone cannot decide where Rachel's
    missiles land (the targeter requires enemy-cluster knowledge that's not
    modeled). When no Counter is pinned, the SCOP fires but applies no
    damage — falling back to global -90 HP would massively over-damage."""
    state = _make_state()
    enemies = [
        _spawn(state, UnitType.TANK, player=1, pos=(5, 5), hp=100),
        _spawn(state, UnitType.INFANTRY, player=1, pos=(10, 10), hp=100),
    ]
    co = state.co_states[state.active_player]
    co.power_bar = co._scop_threshold
    state._oracle_power_aoe_positions = None
    state.step(Action(ActionType.ACTIVATE_SCOP))
    for u in enemies:
        assert u.hp == 100, (
            f"no-pin SCOP must not damage {u.unit_type.name} at {u.pos} "
            f"(got hp={u.hp})"
        )


# ---------------------------------------------------------------------------
# 7. Bonus invariant: overlapping missiles stack damage (multiplicity).
#
# Drilled on gid 1622501 env 20 — missileCoords = [(11,20), (4,9), (11,20)]
# pin Counter[(11,20)]=2, so a unit at (11,20) takes 60 HP, not 30.
# This is the chief reason we widened the channel from set to Counter.
# ---------------------------------------------------------------------------

def test_overlapping_missiles_stack_damage():
    """Two missiles aimed at the same tile damage a unit there for 60 HP."""
    state = _make_state()
    enemy = _spawn(state, UnitType.TANK, player=1, pos=(11, 20), hp=100)
    _fire_covering_fire(state, centers=[(11, 20), (4, 9), (11, 20)])
    assert enemy.hp == 40, f"expected 40 HP after 2 missiles, got {enemy.hp}"
