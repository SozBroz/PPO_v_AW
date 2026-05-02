"""Commander Wars NormalAI-style capture advisor.

This is not a C++ bridge. It converts the useful shape of Commander Wars'
NormalAI capture rules into a deterministic Python heuristic:

- prefer continuing/finishing captures;
- prefer enemy properties over neutral properties;
- prefer HQ / production buildings over ordinary cities;
- prefer nearer captures;
- penalize obviously dangerous exposed capture tiles.

The advisor is intentionally cheap because it runs inside legal-action
generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from engine.terrain import get_terrain
from engine.unit import UNIT_STATS, Unit


Coord = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CaptureScore:
    pos: Coord
    score: float
    owner_bonus: float
    terrain_bonus: float
    continue_bonus: float
    distance_penalty: float
    risk_penalty: float
    nearby_enemy_pressure: int


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _iter_live_enemy_units(state, player: int):
    for p, units in getattr(state, "units", {}).items():
        if int(p) == int(player):
            continue
        for u in units:
            if getattr(u, "is_alive", True):
                yield u


def _property_at(state, pos: Coord):
    try:
        return state.get_property_at(int(pos[0]), int(pos[1]))
    except Exception:
        return None


def _terrain_at(state, pos: Coord):
    tid = state.map_data.terrain[int(pos[0])][int(pos[1])]
    return get_terrain(tid)


def _is_capturable_property(state, player: int, pos: Coord) -> bool:
    terrain = _terrain_at(state, pos)
    if not terrain.is_property:
        return False

    prop = _property_at(state, pos)
    if prop is None:
        return False

    if getattr(prop, "is_comm_tower", False) or getattr(prop, "is_lab", False):
        return False

    owner = getattr(prop, "owner", None)
    return owner is None or int(owner) != int(player)


def _terrain_priority(state, pos: Coord) -> float:
    terrain = _terrain_at(state, pos)
    bonus = 0.0

    if getattr(terrain, "is_hq", False):
        bonus += 2400.0
    if getattr(terrain, "is_base", False):
        bonus += 900.0
    if getattr(terrain, "is_airport", False):
        bonus += 850.0
    if getattr(terrain, "is_port", False):
        bonus += 750.0

    if bonus <= 0.0:
        bonus += 250.0

    return bonus


def _owner_priority(state, player: int, pos: Coord) -> float:
    prop = _property_at(state, pos)
    if prop is None:
        return 0.0

    owner = getattr(prop, "owner", None)
    if owner is None:
        return 500.0
    if int(owner) != int(player):
        return 1100.0

    return -999999.0


def _same_tile_continue_bonus(unit: Unit, pos: Coord) -> float:
    if getattr(unit, "pos", None) != pos:
        return 0.0
    return 1800.0


def _nearby_enemy_pressure(state, player: int, pos: Coord, radius: int = 3) -> int:
    pressure = 0
    for enemy in _iter_live_enemy_units(state, player):
        epos = getattr(enemy, "pos", None)
        if epos is None:
            continue
        d = _manhattan(pos, epos)
        if d <= radius:
            pressure += max(1, radius + 1 - d)
    return pressure


def _risk_penalty(state, player: int, unit: Unit, pos: Coord) -> float:
    pressure = _nearby_enemy_pressure(state, player, pos, radius=3)

    # Your engine appears to use AWBW-style HP as 0..100 in most places.
    # Convert to displayed HP-ish scale for a stable low-HP multiplier.
    raw_hp = float(getattr(unit, "hp", 100) or 100)
    display_hp = raw_hp / 10.0 if raw_hp > 10.0 else raw_hp

    low_hp_multiplier = 1.0 + max(0.0, 7.0 - display_hp) * 0.20
    return 115.0 * pressure * low_hp_multiplier


def score_capture_destination(state, player: int, unit: Unit, pos: Coord) -> CaptureScore | None:
    if not UNIT_STATS[unit.unit_type].can_capture:
        return None
    if not _is_capturable_property(state, player, pos):
        return None

    owner_bonus = _owner_priority(state, player, pos)
    terrain_bonus = _terrain_priority(state, pos)
    continue_bonus = _same_tile_continue_bonus(unit, pos)

    dist = _manhattan(getattr(unit, "pos", pos), pos)
    distance_penalty = 65.0 * float(dist)

    risk = _risk_penalty(state, player, unit, pos)
    pressure = _nearby_enemy_pressure(state, player, pos, radius=3)

    score = 1000.0 + owner_bonus + terrain_bonus + continue_bonus - distance_penalty - risk

    return CaptureScore(
        pos=pos,
        score=score,
        owner_bonus=owner_bonus,
        terrain_bonus=terrain_bonus,
        continue_bonus=continue_bonus,
        distance_penalty=distance_penalty,
        risk_penalty=risk,
        nearby_enemy_pressure=pressure,
    )


def ranked_capture_destinations(
    state,
    player: int,
    unit: Unit,
    positions: Iterable[Coord],
) -> list[CaptureScore]:
    scored: list[CaptureScore] = []
    for pos in positions:
        item = score_capture_destination(state, player, unit, pos)
        if item is not None:
            scored.append(item)

    scored.sort(key=lambda x: (-x.score, x.pos[0], x.pos[1]))
    return scored


def advisor_capture_destinations(
    state,
    player: int,
    unit: Unit,
    reachable: Iterable[Coord],
    *,
    top_k: int = 1,
    score_margin: float = 0.0,
) -> set[Coord]:
    ranked = ranked_capture_destinations(state, player, unit, reachable)
    if not ranked:
        return set()

    best = ranked[0].score
    keep: set[Coord] = set()

    for i, item in enumerate(ranked):
        if i < int(max(1, top_k)) or item.score >= best - float(score_margin):
            keep.add(item.pos)
        else:
            break

    return keep
