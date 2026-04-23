"""Phase 2c correctness gate: per-call occupancy dict must match get_unit_at semantics."""

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
MAP_123858 = 123858

_PLAIN = 1
_SHOAL = 29
_SEA = 28


def _compute_reachable_costs_no_occupancy_cache(
    state: GameState, unit: Unit
) -> dict[tuple[int, int], int]:
    """Reference: same as production including Phase 2b ``_cached_cost``, but ``get_unit_at`` for occupancy."""
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

    _cost_cache: dict[int, int] = {}

    def _cached_cost(tid: int) -> int:
        c = _cost_cache.get(tid)
        if c is None:
            c = effective_move_cost(state, unit, tid)
            _cost_cache[tid] = c
        return c

    while queue:
        (r, c), fuel_used = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < state.map_data.height and 0 <= nc < state.map_data.width):
                continue
            tid = state.map_data.terrain[nr][nc]
            cost = _cached_cost(tid)
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
    unit_type: UnitType,
    player: int,
    pos: tuple[int, int],
    unit_id: int,
    *,
    hp: int = 100,
    loaded_units: list | None = None,
) -> Unit:
    s = UNIT_STATS[unit_type]
    u = Unit(
        unit_type,
        player,
        hp,
        s.max_ammo if s.max_ammo > 0 else 0,
        s.max_fuel,
        pos,
        False,
        [] if loaded_units is None else loaded_units,
        False,
        20,
        unit_id=unit_id,
    )
    return u


def _rect_map(map_id: int, name: str, h: int, w: int, tid: int) -> MapData:
    terrain = [[tid for _ in range(w)] for _ in range(h)]
    return MapData(
        map_id=map_id,
        name=name,
        map_type="std",
        terrain=terrain,
        height=h,
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


def _assert_no_duplicate_positions(states: list[GameState], label: str) -> None:
    for st in states:
        seen: dict[tuple[int, int], int] = {}
        for pl, lst in st.units.items():
            for u in lst:
                if not u.is_alive:
                    continue
                p = u.pos
                if p in seen and seen[p] != u.unit_id:
                    raise AssertionError(
                        f"{label}: two alive units share {p}: ids {seen[p]} and {u.unit_id}"
                    )
                seen[p] = u.unit_id


def _scenario_states() -> list[tuple[GameState, Unit, str]]:
    out: list[tuple[GameState, Unit, str]] = []
    uid = 1

    # Empty board: one unit, wide plains
    wide = _rect_map(991_001, "wide_plain", 1, 24, _PLAIN)
    st_w = make_initial_state(wide, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_w.units = {0: [], 1: []}
    lone = _make_unit(UnitType.INFANTRY, 0, (0, 0), uid)
    uid += 1
    st_w.units[0].append(lone)
    out.append((st_w, lone, "empty_board_wide"))

    # Crowded 5x5 plains: transport load (APC empty), full APC blocks second load, join, blockers
    p5 = _rect_map(991_002, "crowded5", 5, 5, _PLAIN)
    st_c = make_initial_state(p5, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_c.units = {0: [], 1: []}
    # Mover infantry center
    mover = _make_unit(UnitType.INFANTRY, 0, (2, 2), uid)
    uid += 1
    # Empty APC east — can load
    apc_empty = _make_unit(UnitType.APC, 0, (2, 3), uid)
    uid += 1
    # APC north with full cargo — mover cannot stop to load (cargo only in hold, not in units[])
    cargo_inf = _make_unit(UnitType.INFANTRY, 0, (1, 2), uid)
    uid += 1
    apc_full = _make_unit(UnitType.APC, 0, (1, 2), uid, loaded_units=[cargo_inf])
    # Join partners southeast (both injured)
    inf_a = _make_unit(UnitType.INFANTRY, 0, (2, 4), uid, hp=50)
    uid += 1
    inf_b = _make_unit(UnitType.INFANTRY, 0, (3, 4), uid, hp=55)
    uid += 1
    # Enemy blocking west tile — cannot pass through (1,2) is apc_full... put enemy at (2,1)
    enemy_blk = _make_unit(UnitType.TANK, 1, (2, 1), uid)
    uid += 1
    # Enemy on corner (4,4) — may be visited but must not appear as stop if only enemy there
    enemy_corner = _make_unit(UnitType.INFANTRY, 1, (4, 4), uid)
    uid += 1
    # Friendly med tank (3,2) — infantry cannot stop unless join/load (wrong type)
    friend_mt = _make_unit(UnitType.MED_TANK, 0, (3, 2), uid)
    uid += 1
    st_c.units[0].extend(
        [mover, apc_empty, apc_full, inf_a, inf_b, friend_mt]
    )
    st_c.units[1].extend([enemy_blk, enemy_corner])
    out.append((st_c, mover, "crowded5_mover"))

    # Join-focused: mover joins injured ally
    st_j = make_initial_state(p5, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_j.units = {0: [], 1: []}
    mj = _make_unit(UnitType.INFANTRY, 0, (2, 2), uid, hp=40)
    uid += 1
    ally = _make_unit(UnitType.INFANTRY, 0, (2, 3), uid, hp=60)
    uid += 1
    st_j.units[0].extend([mj, ally])
    out.append((st_j, mj, "join_pair"))

    # Lander on shoal with one cargo; second infantry loads from adjacent shoal
    shoal3 = _rect_map(991_003, "shoal3", 3, 3, _SHOAL)
    st_l = make_initial_state(shoal3, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_l.units = {0: [], 1: []}
    cargo2 = _make_unit(UnitType.INFANTRY, 0, (1, 1), uid)
    uid += 1
    lander = _make_unit(UnitType.LANDER, 0, (1, 1), uid, loaded_units=[cargo2])
    uid += 1
    minf = _make_unit(UnitType.INFANTRY, 0, (1, 0), uid)
    uid += 1
    st_l.units[0].extend([lander, minf])
    out.append((st_l, minf, "lander_shoal_load"))

    # T-Copter empty + mech loads (copter capacity 1)
    st_tc = make_initial_state(p5, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_tc.units = {0: [], 1: []}
    tc = _make_unit(UnitType.T_COPTER, 0, (2, 4), uid)
    uid += 1
    mech_m = _make_unit(UnitType.MECH, 0, (2, 2), uid)
    uid += 1
    st_tc.units[0].extend([tc, mech_m])
    out.append((st_tc, mech_m, "tcopter_mech_load"))

    # Andy SCOP +1 move
    plain6 = MapData(
        map_id=991_004,
        name="andy_scop_occ",
        map_type="std",
        terrain=[[_PLAIN] * 8],
        height=1,
        width=8,
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
    st_a = make_initial_state(plain6, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    st_a.units = {0: [], 1: []}
    ia = _make_unit(UnitType.INFANTRY, 0, (0, 0), uid)
    uid += 1
    st_a.units[0].append(ia)
    st_a.co_states[0].scop_active = True
    out.append((st_a, ia, "andy_scop_inf"))

    # Naval 3x3 sea: mover cruiser center, adjacent friendly cruiser
    sea_terrain = [[_SEA] * 3, [_SEA] * 3, [_SEA] * 3]
    sea_md = MapData(
        map_id=991_005,
        name="sea_occ",
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
    cr_m = _make_unit(UnitType.CRUISER, 0, (1, 1), uid)
    uid += 1
    cr_f = _make_unit(UnitType.CRUISER, 0, (1, 0), uid)
    uid += 1
    st_sea.units[0].extend([cr_m, cr_f])
    out.append((st_sea, cr_m, "cruiser_adjacent_cruiser"))

    # Real map (fallback if one id missing)
    for mid, tag in ((MAP_166877, "166877"), (MAP_123858, "123858")):
        try:
            md = load_map(mid, MAP_POOL, MAPS_DIR)
        except Exception:
            continue
        pos = (2, min(17, md.width - 1))
        if pos[0] >= md.height:
            pos = (0, 0)
        st_m = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2", replay_first_mover=0)
        st_m.units = {0: [], 1: []}
        u = _make_unit(UnitType.INFANTRY, 0, pos, uid)
        uid += 1
        st_m.units[0].append(u)
        out.append((st_m, u, f"real_map_{tag}"))

    return out


def _corpus_states() -> list[GameState]:
    pkls = sorted(CORPUS_DIR.glob("*.pkl"))
    out: list[GameState] = []
    for p in pkls:
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


def test_compute_reachable_costs_occupancy_matches_get_unit_at_reference() -> None:
    for state, unit, label in _scenario_states():
        got = compute_reachable_costs(state, unit)
        ref = _compute_reachable_costs_no_occupancy_cache(state, unit)
        assert got == ref, f"{label}: occupancy != get_unit_at: {got!r} vs {ref!r}"

    for state in _corpus_states():
        for player in (0, 1):
            for unit in list(state.units[player]):
                got = compute_reachable_costs(state, unit)
                ref = _compute_reachable_costs_no_occupancy_cache(state, unit)
                uid = getattr(unit, "unit_id", "?")
                assert got == ref, f"corpus p{player} unit_id={uid}: {got!r} vs {ref!r}"


def test_two_units_at_same_pos_invariant() -> None:
    states: list[GameState] = [s for s, _, _ in _scenario_states()]
    states.extend(_corpus_states())
    _assert_no_duplicate_positions(states, "scenario+corpus")
