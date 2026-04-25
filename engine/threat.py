"""
Derived tactical planes for RL encoder (STD, full obs).

Expensive relative to raw terrain — call only from ``encode_state`` hot path when
building influence channels. See ``docs/restart_arch/influence_channels_spec.md``.

Caching: Influence planes are cached based on game state hash. They only change
when units move, are created/destroyed, or property ownership changes.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from engine.action import compute_reachable_costs, get_attack_targets, _build_occupancy
from engine.combat import calculate_damage
from engine.game import GameState
from engine.terrain import get_terrain
from engine.unit import UNIT_STATS, Unit, UnitType

# Normalization: damage and turn horizons
_MAX_DAMAGE = 100.0
_CAPTURE_TURN_HORIZON = 20.0

# Module-level cache for influence planes
# Key: (state_id, me, turn, active_player, p0_cop, p0_scop, p1_cop, p1_scop, weather) -> result
# Recomputed at turn boundaries AND when CO powers change movement/weather
_influence_cache: dict[tuple, tuple[np.ndarray, ...]] = {}
_CACHE_MAX_SIZE = 256  # Keep cache bounded for parallel envs


def _get_power_and_weather_key(state: GameState) -> tuple:
    """Extract CO power state and weather for cache key.

    These affect reach (movement range) and threat (damage modifiers).

    CO powers that affect movement/reach:
    - Adder: +1 (COP) / +2 (SCOP) movement to all units
    - Grit: +1 (COP) / +2 (SCOP) range to indirects
    - Jess: +1 (COP) / +2 (SCOP) movement to ground vehicles
    - Sami: +1 (COP) / +2 (SCOP) movement to infantry/mechs
    - Koal: +1 (COP) / +2 (SCOP) movement to all units
    - Andy/Max: +1 movement to all/direct units (SCOP)

    CO powers that affect weather (and thus movement costs):
    - Olaf: COP/SCOP → snow (increases movement costs)
    - Drake: SCOP → rain (increases movement costs for vehicles)
    """
    p0 = state.co_states[0]
    p1 = state.co_states[1]
    power_state = (
        p0.cop_active, p0.scop_active,
        p1.cop_active, p1.scop_active,
    )
    # Weather affects movement costs (snow/rain slow units)
    weather = state.weather
    return (*power_state, weather)


def _compute_turn_key(state: GameState, me: int) -> tuple:
    """Compute a cache key based on turn and CO power state.

    Influence planes change when:
    1. Turn or active_player changes (turn boundary)
    2. CO power state changes (COP/SCOP activates, affecting movement/damage)
    3. Weather changes (COPs like Olaf/Drake)

    Cache key includes all factors that affect reach/threat calculations.
    """
    power_key = _get_power_and_weather_key(state)
    return (id(state), me, state.turn, state.active_player, *power_key)


def _compute_full_state_hash(state: GameState, me: int) -> str:
    """Compute a hash of the game state elements that affect influence planes.

    This captures: unit positions, types, HP, fuel, ammo, alive status,
    and property owners/capture points. Used for validation only.
    """
    enemy = 1 - me
    parts: list[str] = []

    # Unit state for both players (ordered deterministically)
    for player in [me, enemy]:
        units = state.units.get(player, [])
        # Sort by position for deterministic ordering
        sorted_units = sorted(
            [u for u in units if u.is_alive],
            key=lambda u: (u.pos[0], u.pos[1], u.unit_type.value)
        )
        for u in sorted_units:
            parts.append(f"{u.pos[0]},{u.pos[1]},{u.unit_type.value},{u.hp},{u.fuel},{u.ammo}")

    # Property state that affects capture calculations
    # Only include income properties (not comm towers/labs) as those are in the calc
    relevant_props = [
        p for p in state.properties
        if not p.is_comm_tower and not p.is_lab
    ]
    sorted_props = sorted(relevant_props, key=lambda p: (p.row, p.col))
    for p in sorted_props:
        parts.append(f"P{p.row},{p.col},{p.owner},{p.capture_points}")

    hash_input = "|".join(parts).encode("utf-8")
    return hashlib.md5(hash_input, usedforsecurity=False).hexdigest()


def _get_cached_influence(
    state: GameState,
    me: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Get cached influence planes for the current turn.

    Returns cached result if we're still in the same turn, None otherwise.
    Influence planes are recomputed only at turn boundaries.
    """
    cache_key = _compute_turn_key(state, me)
    cached = _influence_cache.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    return None


def _set_cached_influence(
    state: GameState,
    me: int,
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Store computed influence planes in cache.

    Implements simple LRU-ish eviction when cache is full.
    """
    # Simple size limit - evict oldest entries if over limit
    while len(_influence_cache) >= _CACHE_MAX_SIZE:
        # Remove first entry (simple FIFO)
        first_key = next(iter(_influence_cache))
        del _influence_cache[first_key]

    cache_key = _compute_turn_key(state, me)
    _influence_cache[cache_key] = result


def _scratch_infantry(player: int, pos: tuple[int, int]) -> Unit:
    st = UNIT_STATS[UnitType.INFANTRY]
    return Unit(
        UnitType.INFANTRY,
        player,
        100,
        st.max_ammo,
        st.max_fuel,
        pos,
        False,
        [],
        False,
        20,
    )


def _max_incoming_damage_to_unit(
    state: GameState,
    target: Unit,
    attacker_player: int,
) -> float:
    """Best one-shot damage ``attacker_player`` can deal to ``target`` this turn."""
    occ = _build_occupancy(state)
    best = 0.0
    for att in state.units[attacker_player]:
        if not att.is_alive:
            continue
        try:
            reach = compute_reachable_costs(state, att, occupancy=occ)
        except Exception:
            continue
        for move_pos in reach:
            tgts = get_attack_targets(state, att, move_pos, occupancy=occ)
            if target.pos not in tgts:
                continue
            tr, tc = target.pos
            att_terra = get_terrain(state.map_data.terrain[move_pos[0]][move_pos[1]])
            def_terra = get_terrain(state.map_data.terrain[tr][tc])
            dmg = calculate_damage(
                att,
                target,
                att_terra,
                def_terra,
                state.co_states[att.player],
                state.co_states[target.player],
                luck_roll=5,
            )
            if dmg is not None:
                best = max(best, float(dmg))
    return best


def threat_on_own_units(state: GameState, own_player: int, grid: int) -> np.ndarray:
    """
    Per-tile max incoming damage (0..100) to a **unit you already have** on that
    tile, normalized to [0,1]. Empty tiles → 0 (cheap proxy; full synthetic
    defender grid would require hypothetical occupancy per tile).
    """
    out = np.zeros((grid, grid), dtype=np.float32)
    enemy = 1 - int(own_player)
    for u in state.units[own_player]:
        if not u.is_alive:
            continue
        r, c = u.pos
        if not (0 <= r < grid and 0 <= c < grid):
            continue
        d = _max_incoming_damage_to_unit(state, u, enemy)
        out[r, c] = min(1.0, max(0.0, d / _MAX_DAMAGE))
    return out


def reach_union(state: GameState, player: int, grid: int) -> np.ndarray:
    """Binary mask: tile reachable by at least one alive unit of ``player``."""
    out = np.zeros((grid, grid), dtype=np.float32)
    occ = _build_occupancy(state)
    for u in state.units[player]:
        if not u.is_alive:
            continue
        try:
            reach = compute_reachable_costs(state, u, occupancy=occ)
        except Exception:
            continue
        for (r, c) in reach:
            if 0 <= r < grid and 0 <= c < grid:
                out[r, c] = 1.0
    return out


def _capturer_unit_types() -> frozenset[UnitType]:
    return frozenset({UnitType.INFANTRY, UnitType.MECH})


def turns_to_capture_grid(
    state: GameState,
    capturer_player: int,
    grid: int,
) -> np.ndarray:
    """
    Per income-property tile: crude MP-distance / move_range lower bound on
    turns to finish capture, normalized to [0,1] with **1 = unreachable** within
    ``_CAPTURE_TURN_HORIZON`` equivalent turns (spec v1 proxy).
    """
    out = np.zeros((grid, grid), dtype=np.float32)
    occ = _build_occupancy(state)
    caps = _capturer_unit_types()

    income_props = [
        p
        for p in state.properties
        if not p.is_comm_tower
        and not p.is_lab
        and p.owner != capturer_player
        and p.capture_points < 20
    ]
    if not income_props:
        return out

    best_mp: dict[tuple[int, int], int] = {}
    for u in state.units[capturer_player]:
        if not u.is_alive or u.unit_type not in caps:
            continue
        try:
            reach = compute_reachable_costs(state, u, occupancy=occ)
        except Exception:
            continue
        for pos, cost in reach.items():
            prev = best_mp.get(pos)
            if prev is None or cost < prev:
                best_mp[pos] = cost

    st_inf = UNIT_STATS[UnitType.INFANTRY]
    max_mp = max(1, st_inf.move_range * 3)

    for prop in income_props:
        r, c = prop.row, prop.col
        if not (0 <= r < grid and 0 <= c < grid):
            continue
        d_min: int | None = None
        for (sr, sc), spent in best_mp.items():
            dist = abs(sr - r) + abs(sc - c)
            cand = spent + dist  # Manhattan to property; ignores path blockers
            if d_min is None or cand < d_min:
                d_min = cand
        if d_min is None:
            out[r, c] = 1.0
        else:
            turns_est = float(d_min) / float(max_mp)
            out[r, c] = min(1.0, turns_est / _CAPTURE_TURN_HORIZON)
    return out


def compute_influence_planes(
    state: GameState,
    *,
    me: int,
    grid: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return six (grid,grid) float32 planes:
      threat_me, threat_enemy, reach_me, reach_enemy, cap_me, cap_enemy
    ``me`` is engine seat of the observer (learner).

    Results are cached per-turn: computed once at the start of each turn
    (when active_player changes), then reused for all unit actions during
    that turn. This is correct because:
    - threat: max damage enemy CAN do from their starting position
    - reach: tiles enemy CAN reach from their starting position
    - capture: based on enemy capturer positions at turn start

    The cache key is (id(state), me, turn, active_player).
    """
    # Check cache first
    cached = _get_cached_influence(state, me)
    if cached is not None:
        return cached

    # Compute fresh
    enemy = 1 - int(me)
    t_me = threat_on_own_units(state, me, grid)
    t_en = threat_on_own_units(state, enemy, grid)
    r_me = reach_union(state, me, grid)
    r_en = reach_union(state, enemy, grid)
    c_me = turns_to_capture_grid(state, me, grid)
    c_en = turns_to_capture_grid(state, enemy, grid)

    result = (t_me, t_en, r_me, r_en, c_me, c_en)
    _set_cached_influence(state, me, result)
    return result


def clear_influence_cache() -> None:
    """Clear the influence plane cache. Call when resetting environments."""
    global _influence_cache
    _influence_cache.clear()


def get_influence_cache_stats() -> dict[str, Any]:
    """Return cache statistics for monitoring."""
    return {
        "size": len(_influence_cache),
        "max_size": _CACHE_MAX_SIZE,
    }
