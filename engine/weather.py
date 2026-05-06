"""
AWBW weather movement cost tables and CO immunity rules.

Sources:
  https://awbw.fandom.com/wiki/Weather
  https://awbw.fandom.com/wiki/Olaf
  https://awbw.fandom.com/wiki/Drake

Weather affects movement costs but NOT fog (fog is not used in ranked std play).

This module exposes a single function: effective_move_cost(state, unit, terrain_id)
which replaces the bare get_move_cost() call for all BFS and reachability paths
so that active weather is correctly applied to movement.
"""
from __future__ import annotations

from engine.terrain import (
    get_move_cost, get_terrain,
    MOVE_INF, MOVE_MECH, MOVE_TREAD, MOVE_TIRE_A, MOVE_TIRE_B,
    MOVE_AIR, MOVE_SEA, MOVE_LANDER, MOVE_PIPELINE,
    INF_PASSABLE,
)

# ---------------------------------------------------------------------------
# AWBW wiki movement tables for Rain and Snow
#
# Maps (terrain_category, move_type) → cost.
# Missing entries = impassable (same as clear; weather doesn't open new paths).
#
# Terrain categories used here:
#   plain, wood, road, mountain, river, shoal, sea, reef, pipe,
#   property (city/base*), port
#   (* air cost on base is 2 in snow; otherwise treated like road/property)
#
# Note: "property" means cities / HQ / labs / comm towers / bases (ground entry).
# "port" is handled separately because sea/lander costs change in snow.
# ---------------------------------------------------------------------------

# Rain: Treads/Tires +1 on Plains and Woods only.
_RAIN: dict[tuple[str, str], int] = {
    # Plains
    ("plain", MOVE_INF):    1,
    ("plain", MOVE_MECH):   1,
    ("plain", MOVE_TREAD):  2,
    ("plain", MOVE_TIRE_A): 3,
    ("plain", MOVE_TIRE_B): 3,
    ("plain", MOVE_AIR):    1,
    # Woods
    ("wood", MOVE_INF):     1,
    ("wood", MOVE_MECH):    1,
    ("wood", MOVE_TREAD):   3,
    ("wood", MOVE_TIRE_A):  4,
    ("wood", MOVE_TIRE_B):  4,
    ("wood", MOVE_AIR):     1,
    # Road (unchanged from clear)
    ("road", MOVE_INF):     1,
    ("road", MOVE_MECH):    1,
    ("road", MOVE_TREAD):   1,
    ("road", MOVE_TIRE_A):  1,
    ("road", MOVE_TIRE_B):  1,
    ("road", MOVE_AIR):     1,
    # Mountain (unchanged from clear)
    ("mountain", MOVE_INF):  2,
    ("mountain", MOVE_MECH): 1,
    ("mountain", MOVE_AIR):  1,
    # River (unchanged from clear)
    ("river", MOVE_INF):    2,
    ("river", MOVE_MECH):   1,
    ("river", MOVE_AIR):    1,
    # Shoal (unchanged from clear)
    ("shoal", MOVE_INF):    1,
    ("shoal", MOVE_MECH):   1,
    ("shoal", MOVE_TREAD):  1,
    ("shoal", MOVE_TIRE_A): 1,
    ("shoal", MOVE_TIRE_B): 1,
    ("shoal", MOVE_AIR):    1,
    ("shoal", MOVE_LANDER): 1,
    # Sea (unchanged from clear)
    ("sea", MOVE_SEA):      1,
    ("sea", MOVE_LANDER):   1,
    ("sea", MOVE_AIR):      1,
    # Reef (unchanged from clear)
    ("reef", MOVE_SEA):     2,
    ("reef", MOVE_LANDER):  2,
    ("reef", MOVE_AIR):     1,
    # Property/city/base ground entry (unchanged from clear)
    ("property", MOVE_INF):    1,
    ("property", MOVE_MECH):   1,
    ("property", MOVE_TREAD):  1,
    ("property", MOVE_TIRE_A): 1,
    ("property", MOVE_TIRE_B): 1,
    ("property", MOVE_AIR):    1,
    # Port (unchanged from clear)
    ("port", MOVE_INF):    1,
    ("port", MOVE_MECH):   1,
    ("port", MOVE_TREAD):  1,
    ("port", MOVE_TIRE_A): 1,
    ("port", MOVE_TIRE_B): 1,
    ("port", MOVE_SEA):    1,
    ("port", MOVE_LANDER): 1,
    ("port", MOVE_AIR):    1,
    # Pipe (unchanged from clear)
    ("pipe", MOVE_PIPELINE): 1,
}

# Snow: broader penalties (infantry/mech/air/sea all affected on various tiles).
_SNOW: dict[tuple[str, str], int] = {
    # Plains: infantry×2, tread+1, tires+1, air×2
    ("plain", MOVE_INF):    2,
    ("plain", MOVE_MECH):   1,   # mech unchanged on plains
    ("plain", MOVE_TREAD):  2,
    ("plain", MOVE_TIRE_A): 3,
    ("plain", MOVE_TIRE_B): 3,
    ("plain", MOVE_AIR):    2,
    # Woods: infantry×2, mech unchanged, tread/tires unchanged (already expensive), air×2
    ("wood", MOVE_INF):     2,
    ("wood", MOVE_MECH):    1,
    ("wood", MOVE_TREAD):   2,
    ("wood", MOVE_TIRE_A):  3,
    ("wood", MOVE_TIRE_B):  3,
    ("wood", MOVE_AIR):     2,
    # Road: ground unchanged, air×2
    ("road", MOVE_INF):     1,
    ("road", MOVE_MECH):    1,
    ("road", MOVE_TREAD):   1,
    ("road", MOVE_TIRE_A):  1,
    ("road", MOVE_TIRE_B):  1,
    ("road", MOVE_AIR):     2,
    # Mountain: infantry cost 4 (×2 from clear 2), mech ×2 (1→2), air×2 (1→2)
    ("mountain", MOVE_INF):  4,
    ("mountain", MOVE_MECH): 2,
    ("mountain", MOVE_AIR):  2,
    # River: infantry unchanged (2), mech unchanged (1), air×2
    ("river", MOVE_INF):    2,
    ("river", MOVE_MECH):   1,
    ("river", MOVE_AIR):    2,
    # Shoal: ground unchanged, air×2, lander unchanged
    ("shoal", MOVE_INF):    1,
    ("shoal", MOVE_MECH):   1,
    ("shoal", MOVE_TREAD):  1,
    ("shoal", MOVE_TIRE_A): 1,
    ("shoal", MOVE_TIRE_B): 1,
    ("shoal", MOVE_AIR):    2,
    ("shoal", MOVE_LANDER): 1,
    # Sea: sea×2, lander×2, air×2
    ("sea", MOVE_SEA):      2,
    ("sea", MOVE_LANDER):   2,
    ("sea", MOVE_AIR):      2,
    # Reef: sea/lander unchanged (already 2), air×2
    ("reef", MOVE_SEA):     2,
    ("reef", MOVE_LANDER):  2,
    ("reef", MOVE_AIR):     2,
    # Property/city/base ground: unchanged; air×2
    ("property", MOVE_INF):    1,
    ("property", MOVE_MECH):   1,
    ("property", MOVE_TREAD):  1,
    ("property", MOVE_TIRE_A): 1,
    ("property", MOVE_TIRE_B): 1,
    ("property", MOVE_AIR):    2,
    # Port: ground unchanged, sea/lander×2, air×2
    ("port", MOVE_INF):    1,
    ("port", MOVE_MECH):   1,
    ("port", MOVE_TREAD):  1,
    ("port", MOVE_TIRE_A): 1,
    ("port", MOVE_TIRE_B): 1,
    ("port", MOVE_SEA):    2,
    ("port", MOVE_LANDER): 2,
    ("port", MOVE_AIR):    2,
    # Pipe: Piperunner is wholly unaffected by snow per wiki.
    ("pipe", MOVE_PIPELINE): 1,
}

# ---------------------------------------------------------------------------
# Classify a terrain tile into a weather-table row key.
# ---------------------------------------------------------------------------

def _terrain_category(terrain_id: int) -> str:
    """Return the weather-table row key for the given terrain id."""
    info = get_terrain(terrain_id)
    if info.is_port:
        return "port"
    if info.is_property:
        # base/airport/city/hq/lab/comm_tower — all share 'property' move costs
        # (airport treated same as property for ground entry; air cost differs in snow)
        return "property"
    # Non-property terrain by terrain id range / name
    name = info.name.lower() if hasattr(info, "name") else ""
    if terrain_id == 1 or "plain" in name or terrain_id in (115, 116):
        # Broken pipe seams (rubble) behave like plains
        return "plain"
    if terrain_id == 2 or "mountain" in name:
        return "mountain"
    if terrain_id == 3 or "wood" in name:
        return "wood"
    if 4 <= terrain_id <= 14 or "river" in name:
        return "river"
    if 15 <= terrain_id <= 27 or "road" in name or "bridge" in name:
        return "road"
    if terrain_id == 28 or "sea" in name:
        return "sea"
    if 29 <= terrain_id <= 32 or "shoal" in name:
        return "shoal"
    if terrain_id == 33 or "reef" in name:
        return "reef"
    if 101 <= terrain_id <= 114 or "pipe" in name:
        return "pipe"
    # Teleport, misc: fall back to plain-like
    return "plain"


# ---------------------------------------------------------------------------
# CO immunity checks
# ---------------------------------------------------------------------------

def _has_weather_immunity(co_id: int, weather: str) -> bool:
    """Return True if this CO's units ignore terrain movement penalties for weather."""
    return (
        (co_id == 9 and weather == "snow") or   # Olaf: immune to snow
        (co_id == 5 and weather == "rain")       # Drake: immune to rain terrain effects
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def effective_move_cost(state: "GameState", unit: "Unit", terrain_id: int) -> int:
    """Return the actual movement point cost for ``unit`` entering ``terrain_id``.

    Replaces bare ``get_move_cost(terrain_id, move_type)`` everywhere in the
    engine so that active weather is correctly applied.

    - Clear weather → identical to ``get_move_cost``.
    - Rain/Snow     → AWBW wiki tables applied per terrain category and move type.
    - CO immunity   → Olaf's units ignore snow; Drake's units ignore rain.
    - Piperunner    → unaffected by any weather (pipe category always returns 1).
    - Koal COP/SCOP → −1 / −2 movement points per **road** tile (bridges/roads;
      not property tiles), AWBW wiki / in-game parity.
    - Lash COP/SCOP → passable terrain costs **1** MP (AWBW wiki; no effect under
      global **snow** weather).
    """
    from engine.unit import UNIT_STATS
    stats = UNIT_STATS[unit.unit_type]
    move_type = stats.move_type

    # Base cost (clear-weather; also the floor for immune COs)
    base = get_move_cost(terrain_id, move_type)
    if base >= INF_PASSABLE:
        return base  # impassable regardless of weather

    weather = state.weather
    if weather == "clear":
        cost = base
    else:
        # CO immunity: own unit is unaffected by terrain penalties of this weather
        co_id = state.co_states[unit.player].co_id
        if _has_weather_immunity(co_id, weather):
            cost = base
        else:
            table = _RAIN if weather == "rain" else _SNOW
            cat = _terrain_category(terrain_id)
            weather_cost = table.get((cat, move_type))
            if weather_cost is None:
                # No entry in weather table: terrain is still impassable under weather
                return INF_PASSABLE
            cost = weather_cost

    co_state = state.co_states[unit.player]

    # Sturm (co_id=29) D2D: all terrain costs 1 MP, except in Snow.
    # AWBW wiki: "Movement cost over all terrain is reduced to 1, except in Snow."
    if co_state.co_id == 29 and weather != "snow":
        cost = 1

    # Koal (co_id 21) Forced March / Trail of Woe: cheaper road movement.
    if co_state.co_id == 21 and (co_state.cop_active or co_state.scop_active):
        if _terrain_category(terrain_id) == "road":
            road_bonus = 2 if co_state.scop_active else 1
            cost = max(0, cost - road_bonus)

    # Lash (co_id 16) Terrain Tactics / Prime Tactics: AWBW treats all passable
    # tiles as cost 1 during powers (desync_audit engine_illegal_move on Lash
    # mirrors e.g. map 159501). Snow weather disables this flattening (wiki).
    if co_state.co_id == 16 and (co_state.cop_active or co_state.scop_active):
        if weather != "snow":
            cost = 1

    return cost
