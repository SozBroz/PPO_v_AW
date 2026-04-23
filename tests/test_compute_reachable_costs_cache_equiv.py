"""Phase 2b correctness gate. The per-call cache must produce byte-identical reachability dicts; any drift means the cache scope is wrong and would silently corrupt training."""

from __future__ import annotations

import collections
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import (  # noqa: E402
    compute_reachable_costs,
    get_loadable_into,
    units_can_join,
)
from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import MapData, load_map  # noqa: E402
from engine.terrain import INF_PASSABLE  # noqa: E402
from engine.unit import UNIT_STATS, Unit, UnitType  # noqa: E402
from engine.weather import effective_move_cost  # noqa: E402

CORPUS_DIR = ROOT / "tests" / "data" / "legal_actions_equivalence_corpus"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
MAP_166877 = 166877

# HRoad — exercises Koal road discount in effective_move_cost when combined with Koal COP.
_ROAD = 15
_PLAIN = 1
_SEA = 28


def _compute_reachable_costs_uncached(state: GameState, unit: Unit) -> dict[tuple[int, int], int]:
    """Reference: ``compute_reachable_costs`` with direct ``effective_move_cost`` per step (no tid cache)."""
    stats = UNIT_STATS[unit.unit_type]
    move_range = stats.move_range
    co = state.co_states[unit.player]

    if co.co_id == 11:
        move_range += 1
        if co.cop_active:
            move_range += 1
        if co.scop_active:
            move_range += 2

    if co.co_id == 8:
        if stats.unit_class == "infantry":
            if co.scop_active:
                move_range += 2
            elif co.cop_active:
                move_range += 1

    if co.co_id == 20 and co.scop_active:
        if stats.unit_class in ("infantry", "mech", "vehicle", "pipe"):
            move_range += 3

    if co.co_id == 14 and (co.cop_active or co.scop_active):
        if stats.unit_class == "vehicle":
            move_range += 2

    if co.co_id == 1 and co.scop_active:
        move_range += 1

    if co.co_id == 21 and co.cop_active:
        move_range += 1

    move_range = min(move_range, unit.fuel)

    start = unit.pos
    visited: dict[tuple[int, int], int] = {start: 0}
    queue: collections.deque[tuple[tuple[int, int], int]] = collections.deque([(start, 0)])

    while queue:
        (r, c), fuel_used = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < state.map_data.height and 0 <= nc < state.map_data.width):
                continue
            tid = state.map_data.terrain[nr][nc]
            cost = effective_move_cost(state, unit, tid)
            if cost >= INF_PASSABLE:
                continue
            new_fuel = fuel_used + cost
            if new_fuel > move_range:
                continue

            occupant = state.get_unit_at(nr, nc)
            if occupant is not None and occupant.player != unit.player:
                continue

            if (nr, nc) not in visited or visited[(nr, nc)] > new_fuel:
                visited[(nr, nc)] = new_fuel
                queue.append(((nr, nc), new_fuel))

    result: dict[tuple[int, int], int] = {}
    for pos, cost in visited.items():
        occupant = state.get_unit_at(*pos)
        if occupant is None or pos == unit.pos:
            result[pos] = cost
        elif occupant.player == unit.player:
            cap = UNIT_STATS[occupant.unit_type].carry_capacity
            if cap > 0 and unit.unit_type in get_loadable_into(occupant.unit_type):
                if len(occupant.loaded_units) < cap:
                    result[pos] = cost
            elif units_can_join(unit, occupant):
                result[pos] = cost

    return result


def _make_unit(
    unit_type: UnitType, player: int, pos: tuple[int, int], unit_id: int
) -> Unit:
    s = UNIT_STATS[unit_type]
    return Unit(
        unit_type,
        player,
        100,
        s.max_ammo if s.max_ammo > 0 else 0,
        s.max_fuel,
        pos,
        False,
        [],
        False,
        20,
        unit_id=unit_id,
    )


def _strip_map(map_id: int, name: str, terrain_row: list[int]) -> MapData:
    w = len(terrain_row)
    return MapData(
        map_id=map_id,
        name=name,
        map_type="std",
        terrain=[list(terrain_row)],
        height=1,
        width=w,
        cap_limit=999,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )


def _synthetic_pairs() -> list[tuple[GameState, Unit, str]]:
    """Non-trivial (state, unit) pairs: varied move types, CO power affecting MP, road/weather."""
    out: list[tuple[GameState, Unit, str]] = []
    uid = 1

    # Andy SCOP: 1x6 plain, infantry with Hyper Upgrade
    plain6 = _strip_map(990_301, "andy_scop_cache", [1, 1, 1, 1, 1, 1])
    st_andy = make_initial_state(plain6, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_andy.units = {0: [], 1: []}
    inf1 = _make_unit(UnitType.INFANTRY, 0, (0, 0), uid)
    uid += 1
    st_andy.units[0].append(inf1)
    st_andy.co_states[0].scop_active = True
    out.append((st_andy, inf1, "andy_scop_inf"))

    st_andy2 = make_initial_state(plain6, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_andy2.units = {0: [], 1: []}
    inf2 = _make_unit(UnitType.INFANTRY, 0, (0, 0), uid)
    uid += 1
    st_andy2.units[0].append(inf2)
    st_andy2.co_states[0].scop_active = False
    out.append((st_andy2, inf2, "andy_base_inf"))

    # Koal COP on mixed plain/road: exercises tid cache across repeated road terrain ids
    mix = _strip_map(990_302, "koal_road_cache", [_PLAIN, _PLAIN, _ROAD, _ROAD, _PLAIN, _PLAIN])
    st_koal = make_initial_state(mix, 21, 14, starting_funds=0, tier_name="T4", replay_first_mover=0)
    st_koal.units = {0: [], 1: []}
    mech = _make_unit(UnitType.MECH, 0, (0, 0), uid)
    uid += 1
    st_koal.units[0].append(mech)
    st_koal.co_states[0].cop_active = True
    out.append((st_koal, mech, "koal_cop_mech_roads"))

    # Rain weather: Olaf (co 5?) — use default COs; snow/rain table path in effective_move_cost
    rain_map = _strip_map(990_303, "rain_cache", [1, 1, 1, 1])
    st_rain = make_initial_state(
        rain_map, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0, default_weather="rain"
    )
    st_rain.units = {0: [], 1: []}
    inf_r = _make_unit(UnitType.INFANTRY, 0, (0, 0), uid)
    uid += 1
    st_rain.units[0].append(inf_r)
    out.append((st_rain, inf_r, "rain_andy_inf"))

    # 3x3 all sea: cruiser
    sea_terrain = [[_SEA, _SEA, _SEA], [_SEA, _SEA, _SEA], [_SEA, _SEA, _SEA]]
    sea_md = MapData(
        map_id=990_304,
        name="sea_cache",
        map_type="std",
        terrain=sea_terrain,
        height=3,
        width=3,
        cap_limit=999,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )
    st_sea = make_initial_state(sea_md, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_sea.units = {0: [], 1: []}
    cru = _make_unit(UnitType.CRUISER, 0, (1, 1), uid)
    uid += 1
    st_sea.units[0].append(cru)
    out.append((st_sea, cru, "cruiser_sea"))

    # Real map: one fresh state per unit on varied terrain (same staging tile)
    md = load_map(MAP_166877, MAP_POOL, MAPS_DIR)
    pos = (2, 17)
    for ut in (UnitType.INFANTRY, UnitType.MECH, UnitType.B_COPTER, UnitType.MED_TANK):
        st_m = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2", replay_first_mover=0)
        st_m.units = {0: [], 1: []}
        u = _make_unit(ut, 0, pos, uid)
        uid += 1
        st_m.units[0].append(u)
        out.append((st_m, u, f"map166877_{ut.name}"))

    return out


def _corpus_states() -> list[GameState]:
    pkls = sorted(CORPUS_DIR.glob("*.pkl"))
    out: list[GameState] = []
    for p in pkls:
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


def test_compute_reachable_costs_matches_uncached_reference() -> None:
    for state, unit, label in _synthetic_pairs():
        c = compute_reachable_costs(state, unit)
        ref = _compute_reachable_costs_uncached(state, unit)
        assert c == ref, f"{label}: cached != uncached: {c!r} vs {ref!r}"

    for state in _corpus_states():
        for player in (0, 1):
            for unit in list(state.units[player]):
                c = compute_reachable_costs(state, unit)
                ref = _compute_reachable_costs_uncached(state, unit)
                uid = getattr(unit, "unit_id", "?")
                assert c == ref, f"corpus p{player} unit_id={uid}: {c!r} vs {ref!r}"
