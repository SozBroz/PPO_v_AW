"""
Per-unit modifier features for the policy encoder.

These helpers are encoder-only: they read the same engine state and helper
rules used by legal movement and combat, but do not mutate gameplay state.
Values are normalized for spatial planes and may be negative where AWBW rules
allow a penalty (for example Flak/Jugger bad luck or Colin day-to-day attack).
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.action import _effective_move_range_cap
from engine.combat import _colin_atk_rider, _kindle_atk_rider, luck_net_bounds_for_co
from engine.game import GameState
from engine.terrain import get_terrain
from engine.unit import UNIT_STATS, Unit

MOVE_DELTA_NORM = 10.0
ATTACK_DEFENSE_NORM = 100.0
LUCK_NORM = 100.0
RANGE_NORM = 10.0


@dataclass(frozen=True)
class UnitModifierFeatures:
    """Normalized feature values written at an occupied cell."""

    move_delta: float
    attack_delta: float
    defense_delta: float
    luck_min: float
    luck_max: float
    indirect_min_range: float
    indirect_max_range: float


def attack_value_for_unit(state: GameState, unit: Unit) -> int:
    """
    Return the attack value used by combat before HP/luck/defender terms.

    This mirrors ``engine.combat.calculate_damage`` for all attacker-side
    modifiers that are knowable from the unit's current tile: CO total ATK,
    Javier towers, Lash terrain-on-power, Kindle urban rider, and Colin SCOP
    Power of Money.
    """
    co = state.co_states[unit.player]
    terrain = get_terrain(state.map_data.terrain[unit.pos[0]][unit.pos[1]])

    av = co.total_atk_for_unit(unit.unit_type)
    if co.co_id == 16 and (co.scop_active or co.cop_active):
        av += terrain.defense * 10
    av += _kindle_atk_rider(co, terrain)
    av += _colin_atk_rider(co)
    return av


def defense_value_for_unit(state: GameState, unit: Unit) -> int:
    """Return CO/tower defense value; terrain stars remain a separate map plane."""
    co = state.co_states[unit.player]
    return co.total_def_for_unit(unit.unit_type)


def luck_envelope_for_unit(state: GameState, unit: Unit) -> tuple[int, int]:
    """Return min/max net luck contribution ``L - LB`` for the owning CO's current power state."""
    co = state.co_states[unit.player]
    return luck_net_bounds_for_co(co)


def modifier_features_for_unit(state: GameState, unit: Unit) -> UnitModifierFeatures:
    """Compute normalized policy features for ``unit`` at its current tile."""
    stats = UNIT_STATS[unit.unit_type]
    effective_move = _effective_move_range_cap(state, unit)
    luck_min, luck_max = luck_envelope_for_unit(state, unit)

    if stats.is_indirect:
        min_range = stats.min_range / RANGE_NORM
        max_range = (stats.max_range + state.co_states[unit.player].range_modifier_for_unit(unit.unit_type)) / RANGE_NORM
    else:
        min_range = 0.0
        max_range = 0.0

    return UnitModifierFeatures(
        move_delta=(effective_move - stats.move_range) / MOVE_DELTA_NORM,
        attack_delta=(attack_value_for_unit(state, unit) - 100) / ATTACK_DEFENSE_NORM,
        defense_delta=(defense_value_for_unit(state, unit) - 100) / ATTACK_DEFENSE_NORM,
        luck_min=luck_min / LUCK_NORM,
        luck_max=luck_max / LUCK_NORM,
        indirect_min_range=min_range,
        indirect_max_range=max_range,
    )
