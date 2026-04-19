"""
Export a list of GameState turn-start snapshots as an AWBW Replay Player–compatible .zip.

File format:
  <game_id>.zip
  └── <anything>   (gzip-compressed, content starts with  O:8:"awbwGame": )
      one line per player-turn, separated by \\n

The replay file (p: action stream) is intentionally omitted in this version;
the viewer will display turn-by-turn snapshots without per-action animation.
"""
from __future__ import annotations

import copy
import gzip
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.game import GameState, make_initial_state
from engine.map_loader import MapData, load_map
from engine.unit import UnitType, UNIT_STATS
from engine.terrain import TERRAIN_TABLE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AWBW constants
# ---------------------------------------------------------------------------

# Fake stable player IDs (the viewer treats them as long integers)
P0_PLAYER_ID: int = 100001
P1_PLAYER_ID: int = 100002
P0_USERS_ID:  int = 200001
P1_USERS_ID:  int = 200002

# Country IDs assigned to P0/P1 — will be overridden from map data when possible
DEFAULT_P0_COUNTRY = 1  # Orange Star
DEFAULT_P1_COUNTRY = 2  # Blue Moon

# Country → (base terrain_id, layout)
# Early countries: city / base / airport / port / HQ at base+0..4
_EARLY = {1: 38, 2: 43, 3: 48, 4: 53}
# Mid countries: RF=6, GS=7, BH=5, BD=8  (same layout)
_MID   = {6: 81, 7: 86, 5: 91, 8: 96}
# Labs for all "classic" countries live in the 138–148 range (not contiguous
# with the country's other properties). Without this, an owned lab falls
# through to "city" in the _EARLY/_MID branches below and renders as a regular
# city in the viewer, silently inflating the apparent income-property count.
_COUNTRY_LAB: dict[int, int] = {
    9: 138,   # AB
    5: 139,   # BH
    2: 140,   # BM
    8: 141,   # BD
    3: 142,   # GE
    7: 143,   # GS
    10: 144,  # JS
    1: 146,   # OS
    6: 147,   # RF
    4: 148,   # YC
}
_COUNTRY_COMM_TOWER: dict[int, int] = {
    9: 127,   # AB
    5: 128,   # BH
    2: 129,   # BM
    8: 130,   # BD
    3: 131,   # GE
    10: 132,  # JS
    1: 134,   # OS
    6: 135,   # RF
    4: 136,   # YC
    7: 137,   # GS
}
# AB (9): airport/base/city/HQ/port = 117..121
_AB    = {9: 117}
# JS (10): airport/base/city/HQ/port = 122..126
_JS    = {10: 122}
# Extended countries (11-20): airport/base/city/com_tower/HQ/lab/port = base+0..6
_EXT = {
    11: 149, 12: 156, 13: 163, 14: 170,
    15: 181, 16: 188, 17: 196, 18: 203, 19: 210, 20: 217,
}

# Non-property tiles that AWBW still treats as `awbwBuilding` entries in its
# canonical map. Omitting these causes the "Buildings do not match replay"
# warning in the AWBW Replay Player.
#   111 = Missile Silo (loaded)
#   112 = Missile Silo (empty)
#   113 = HPipe Seam
#   114 = VPipe Seam
AWBW_NON_PROPERTY_BUILDING_TIDS: set[int] = {111, 112, 113, 114}


def _awbw_non_property_building_capture(terrain_id: int) -> int:
    """PHP `awbwBuilding.capture` for tiles that are buildings but not properties.

    Pipe seams store **remaining seam HP** in this field (99 at full health), not
    capture progress. Missile silos use 20 per AWBW snapshots. The desktop viewer
    may mislabel seam HP in the UI; match AWBW's numeric field here.
    """
    if terrain_id in (113, 114):
        return 99
    return 20


# ---------------------------------------------------------------------------
# AWBW unit name mapping (must match the viewer's unit database)
# ---------------------------------------------------------------------------
_AWBW_UNIT_NAMES: dict[UnitType, str] = {
    UnitType.INFANTRY:   "Infantry",
    UnitType.MECH:       "Mech",
    UnitType.RECON:      "Recon",
    UnitType.TANK:       "Tank",
    UnitType.MED_TANK:   "Md. Tank",
    UnitType.NEO_TANK:   "Neo Tank",
    UnitType.MEGA_TANK:  "Mega Tank",
    UnitType.APC:        "APC",
    UnitType.ARTILLERY:  "Artillery",
    UnitType.ROCKET:     "Rockets",
    UnitType.ANTI_AIR:   "Anti-Air",
    UnitType.MISSILES:   "Missile",
    UnitType.FIGHTER:    "Fighter",
    UnitType.BOMBER:     "Bomber",
    UnitType.STEALTH:    "Stealth",
    UnitType.B_COPTER:   "B-Copter",
    UnitType.T_COPTER:   "T-Copter",
    UnitType.BATTLESHIP: "Battleship",
    UnitType.CARRIER:    "Carrier",
    UnitType.SUBMARINE:  "Sub",
    UnitType.CRUISER:    "Cruiser",
    UnitType.LANDER:     "Lander",
    UnitType.GUNBOAT:    "Gunboat",
    UnitType.BLACK_BOAT: "Black Boat",
    UnitType.BLACK_BOMB: "Black Bomb",
    UnitType.PIPERUNNER: "Piperunner",
    UnitType.OOZIUM:     "Oozium",
}

# AWBW movement type strings
_COPTER_TYPES = {UnitType.B_COPTER, UnitType.T_COPTER}
_MOVE_TYPE_MAP: dict[str, str] = {
    "infantry": "foot",
    "mech":     "boots",
    "tread":    "treads",
    "tire_a":   "tires",
    "tire_b":   "tires",
    "air":      "air",      # fighters/bombers; copters overridden below
    "sea":      "sea",
    "lander":   "lander",
    "pipe":     "pipeline",
}


def _awbw_move_type(unit_type: UnitType) -> str:
    if unit_type in _COPTER_TYPES:
        return "copter"
    stats = UNIT_STATS[unit_type]
    return _MOVE_TYPE_MAP.get(stats.move_type, stats.move_type)


# ---------------------------------------------------------------------------
# Terrain ID resolution for current owner
# ---------------------------------------------------------------------------

def _terrain_id_for_owner(
    original_tid: int,
    owner: Optional[int],
    player_to_country: dict[int, int],
) -> int:
    """Return the AWBW terrain_id that reflects the current ownership."""
    info = TERRAIN_TABLE.get(original_tid)
    if info is None or not info.is_property:
        return original_tid

    if owner is None:
        # Neutral
        if info.is_base:        return 35
        if info.is_airport:     return 36
        if info.is_port:        return 37
        if info.is_hq:          return 34
        if info.is_lab:         return 145
        if info.is_comm_tower:  return 133
        return 34

    cid = player_to_country.get(owner)
    if cid is None:
        return original_tid  # unknown country — keep original

    if cid in _EARLY:
        base = _EARLY[cid]
        if info.is_hq:          return base + 4
        if info.is_port:        return base + 3
        if info.is_airport:     return base + 2
        if info.is_base:        return base + 1
        if info.is_lab:         return _COUNTRY_LAB[cid]
        if info.is_comm_tower:  return _COUNTRY_COMM_TOWER[cid]
        return base + 0  # city

    if cid in _MID:
        base = _MID[cid]
        if info.is_hq:          return base + 4
        if info.is_port:        return base + 3
        if info.is_airport:     return base + 2
        if info.is_base:        return base + 1
        if info.is_lab:         return _COUNTRY_LAB[cid]
        if info.is_comm_tower:  return _COUNTRY_COMM_TOWER[cid]
        return base + 0

    if cid == 9:  # AB
        if info.is_airport:    return 117
        if info.is_base:       return 118
        if info.is_hq:         return 120
        if info.is_port:       return 121
        if info.is_lab:        return _COUNTRY_LAB[cid]
        if info.is_comm_tower: return _COUNTRY_COMM_TOWER[cid]
        return 119

    if cid == 10:  # JS
        if info.is_airport:    return 122
        if info.is_base:       return 123
        if info.is_hq:         return 125
        if info.is_port:       return 126
        if info.is_lab:        return _COUNTRY_LAB[cid]
        if info.is_comm_tower: return _COUNTRY_COMM_TOWER[cid]
        return 124

    if cid in _EXT:
        base = _EXT[cid]
        if info.is_airport:    return base + 0
        if info.is_base:       return base + 1
        if info.is_comm_tower: return base + 3
        if info.is_hq:         return base + 4
        if info.is_lab:        return base + 5
        if info.is_port:       return base + 6
        return base + 2  # city

    return original_tid  # fallback


# ---------------------------------------------------------------------------
# Weather helpers
# ---------------------------------------------------------------------------

_WEATHER_TYPE: dict[str, str] = {"clear": "Clear", "rain": "Rain", "snow": "Snow"}
_WEATHER_CODE: dict[str, str] = {"clear": "C",     "rain": "R",    "snow": "S"}


def _weather_type_str(state: GameState) -> str:
    return _WEATHER_TYPE.get(getattr(state, "weather", "clear"), "Clear")


def _weather_code_str(state: GameState) -> str:
    return _WEATHER_CODE.get(getattr(state, "weather", "clear"), "C")


# ---------------------------------------------------------------------------
# PHP serialization primitives
# ---------------------------------------------------------------------------

def _php_str(s: str) -> str:
    encoded = s.encode("utf-8")
    n = len(encoded)
    return f's:{n}:"{s}";'


def _php_int(n: int) -> str:
    return f"i:{n};"


def _php_long(n: int) -> str:
    return f"i:{n};"


def _php_null() -> str:
    return "N;"


def _php_float(f: float) -> str:
    # PHP uses 'd' for doubles; format without trailing zeros except at least one decimal
    formatted = f"{f:g}"
    if "." not in formatted and "e" not in formatted:
        formatted += ".0"
    return f"d:{formatted};"


def _php_bool_str(b: bool) -> str:
    """AWBW stores booleans as strings 'Y'/'N'."""
    return _php_str("Y" if b else "N")


def _php_key(s: str) -> str:
    return _php_str(s)


def _php_object(classname: str, fields: list[tuple[str, str]]) -> str:
    clen = len(classname.encode("utf-8"))
    n = len(fields)
    body = "".join(_php_key(k) + v for k, v in fields)
    return f'O:{clen}:"{classname}":{n}:{{{body}}}'


def _php_array(items: list[tuple[str, str]]) -> str:
    """items is [(key_serialized, value_serialized), ...]"""
    n = len(items)
    body = "".join(k + v for k, v in items)
    return f"a:{n}:{{{body}}}"


# ---------------------------------------------------------------------------
# Per-object serializers
# ---------------------------------------------------------------------------

def _serialize_player(
    player_idx: int,
    player_id: int,
    users_id: int,
    team_name: str,
    country_id: int,
    co_id: int,
    funds: int,
    co_power: int,
    co_max_power: Optional[int],
    co_max_spower: Optional[int],
    eliminated: bool,
    co_power_on: str,   # "N", "Y", "S"
    game_id: int,
    order: int,
) -> str:
    fields = [
        ("id",                   _php_long(player_id)),
        ("users_id",             _php_long(users_id)),
        ("team",                 _php_str(str(player_id))),
        ("countries_id",         _php_int(country_id)),
        ("co_id",                _php_int(co_id)),
        ("tags_co_id",           _php_null()),
        ("co_max_power",         _php_int(co_max_power) if co_max_power is not None else _php_null()),
        ("co_max_spower",        _php_int(co_max_spower) if co_max_spower is not None else _php_null()),
        ("tags_co_max_power",    _php_null()),
        ("tags_co_max_spower",   _php_null()),
        ("funds",                _php_int(funds)),
        ("eliminated",           _php_bool_str(eliminated)),
        ("co_power",             _php_int(co_power)),
        ("tags_co_power",        _php_null()),
        ("order",                _php_int(order + 1)),
        ("accept_draw",          _php_bool_str(False)),
        ("co_power_on",          _php_str(co_power_on)),
        ("co_image",             _php_null()),
        ("email",                _php_null()),
        ("emailpress",           _php_null()),
        ("last_read",            _php_null()),
        ("last_read_broadcasts", _php_null()),
        ("games_id",             _php_long(game_id)),
        ("signature",            _php_null()),
        ("turn",                 _php_null()),
        ("turn_start",           _php_null()),
        ("turn_clock",           _php_null()),
        ("aet_count",            _php_int(0)),
        ("uniq_id",              _php_null()),
        ("interface",            _php_null()),
    ]
    return _php_object("awbwPlayer", fields)


def _serialize_building(
    bld_id: int,
    terrain_id: int,
    col: int,     # 0-based
    row: int,     # 0-based
    capture_points: int,
    game_id: int,
) -> str:
    fields = [
        ("id",           _php_long(bld_id)),
        ("terrain_id",   _php_int(terrain_id)),
        ("x",            _php_int(col)),   # viewer uses 0-based grid coords
        ("y",            _php_int(row)),
        ("capture",      _php_int(capture_points)),
        ("last_capture", _php_int(capture_points)),
        ("games_id",     _php_long(game_id)),
        ("last_updated", _php_null()),
    ]
    return _php_object("awbwBuilding", fields)


def _serialize_unit(
    unit_id: int,
    player_id: int,
    unit_type: UnitType,
    col: int,     # 0-based
    row: int,     # 0-based
    hp: int,      # 1-100
    ammo: int,
    fuel: int,
    moved: bool,
    is_submerged: bool,
    game_id: int,
) -> str:
    stats = UNIT_STATS[unit_type]
    # AWBW HP scale: 0.1–10 (e.g. 100 → 10, 55 → 5.5)
    awbw_hp = hp / 10.0

    # Range: min/max attack range
    short_range = stats.min_range if stats.min_range > 0 else 0
    long_range  = stats.max_range if stats.max_range > 0 else 0
    # Unarmed units: both 0
    if stats.max_ammo == 0:
        short_range = 0
        long_range  = 0

    fields = [
        ("id",              _php_long(unit_id)),
        ("players_id",      _php_long(player_id)),
        ("name",            _php_str(_AWBW_UNIT_NAMES.get(unit_type, stats.name))),
        ("movement_points", _php_int(stats.move_range)),
        ("vision",          _php_int(stats.vision)),
        ("fuel",            _php_int(fuel)),
        ("fuel_per_turn",   _php_int(stats.fuel_per_turn)),
        ("sub_dive",        _php_str("Y" if is_submerged else "N")),
        ("ammo",            _php_int(ammo)),
        ("short_range",     _php_int(short_range)),
        ("long_range",      _php_int(long_range)),
        ("second_weapon",   _php_bool_str(False)),
        ("cost",            _php_int(stats.cost)),
        ("movement_type",   _php_str(_awbw_move_type(unit_type))),
        ("x",               _php_int(col)),   # viewer uses 0-based grid coords
        ("y",               _php_int(row)),
        ("moved",           _php_int(1 if moved else 0)),
        ("capture",         _php_int(0)),
        ("fired",           _php_int(0)),
        ("hit_points",      _php_float(awbw_hp)),
        ("cargo1_units_id", _php_long(0)),
        ("cargo2_units_id", _php_long(0)),
        ("carried",         _php_bool_str(False)),
        ("games_id",        _php_long(game_id)),
        ("symbol",          _php_str(_AWBW_UNIT_NAMES.get(unit_type, stats.name))),
    ]
    return _php_object("awbwUnit", fields)


# ---------------------------------------------------------------------------
# Turn snapshot serializer
# ---------------------------------------------------------------------------

def serialize_turn_snapshot(
    state: GameState,
    game_id: int,
    player_ids: tuple[int, int],       # (p0_id, p1_id)
    users_ids: tuple[int, int],        # (p0_uid, p1_uid)
    countries: tuple[int, int],        # (p0_country_id, p1_country_id)
    player_to_country: dict[int, int], # player_index → country_id
    game_name: str,
    start_date: str,
) -> str:
    """Serialize the current GameState as one awbwGame PHP string (one line)."""
    p0_id, p1_id = player_ids
    p0_uid, p1_uid = users_ids
    p0_cid, p1_cid = countries
    active_pid = p0_id if state.active_player == 0 else p1_id

    # ---- Players ----
    def _co_power_on_str(cs) -> str:
        if cs.scop_active: return "S"
        if cs.cop_active:  return "Y"
        return "N"

    def _co_max(cs, cop: bool) -> Optional[int]:
        # AWBW viewer's PowerProgress bar uses ProgressPerBar = 90000 (not 9000).
        # co_max_power/co_max_spower must be exact multiples of 90000 so the integer
        # division smallBars = value / 90000 yields non-zero segments, preventing
        # the DivideByZeroException in onCOChange.
        if cop:
            if cs.cop_stars is None:
                return None
            v = cs.cop_stars * 90000
            return v if v > 0 else None
        total = (cs.cop_stars or 0) + cs.scop_stars
        v2 = total * 90000
        return v2 if v2 > 0 else None

    co0 = state.co_states[0]
    co1 = state.co_states[1]

    player_items = [
        (_php_int(0), _serialize_player(
            0, p0_id, p0_uid, "A", p0_cid, co0.co_id,
            state.funds[0], co0.power_bar,
            _co_max(co0, True), _co_max(co0, False),
            False, _co_power_on_str(co0), game_id, 0,
        )),
        (_php_int(1), _serialize_player(
            1, p1_id, p1_uid, "B", p1_cid, co1.co_id,
            state.funds[1], co1.power_bar,
            _co_max(co1, True), _co_max(co1, False),
            False, _co_power_on_str(co1), game_id, 1,
        )),
    ]
    players_array = _php_array(player_items)

    # ---- Buildings ----
    # Pass 1: capturable properties (cities, bases, HQ, etc.)
    bld_items = []
    occupied: set[tuple[int, int]] = set()
    next_bld_id = 1
    for idx, prop in enumerate(state.properties):
        tid = _terrain_id_for_owner(prop.terrain_id, prop.owner, player_to_country)
        bld_ser = _serialize_building(
            bld_id=next_bld_id,
            terrain_id=tid,
            col=prop.col,
            row=prop.row,
            capture_points=prop.capture_points,
            game_id=game_id,
        )
        bld_items.append((_php_int(len(bld_items)), bld_ser))
        occupied.add((prop.row, prop.col))
        next_bld_id += 1

    # Pass 2: non-property tiles AWBW still treats as buildings
    # (missile silos, pipe seams). Walk the canonical terrain grid.
    terrain_grid = state.map_data.terrain
    for row_idx, row in enumerate(terrain_grid):
        for col_idx, tid in enumerate(row):
            if tid not in AWBW_NON_PROPERTY_BUILDING_TIDS:
                continue
            if (row_idx, col_idx) in occupied:
                continue
            bld_ser = _serialize_building(
                bld_id=next_bld_id,
                terrain_id=tid,
                col=col_idx,
                row=row_idx,
                capture_points=_awbw_non_property_building_capture(tid),
                game_id=game_id,
            )
            bld_items.append((_php_int(len(bld_items)), bld_ser))
            next_bld_id += 1
    buildings_array = _php_array(bld_items)

    # ---- Units ----
    # Use the unit's stable `unit_id` (monotonic, assigned at creation and
    # never reused). AWBW's DrawableUnit is indexed by this id and its
    # `UpdateUnit` does not re-evaluate unit type, so any id churn across
    # snapshots renders the wrong sprite/color for later turns.
    unit_items = []
    fallback_id = 10_000  # only used if a Unit somehow has unit_id == 0
    for player_idx, unit_list in sorted(state.units.items()):
        pid = p0_id if player_idx == 0 else p1_id
        for unit in unit_list:
            uid = unit.unit_id
            if uid <= 0:
                fallback_id += 1
                uid = fallback_id
            unit_ser = _serialize_unit(
                unit_id=uid,
                player_id=pid,
                unit_type=unit.unit_type,
                col=unit.pos[1],
                row=unit.pos[0],
                hp=unit.hp,
                ammo=unit.ammo,
                fuel=unit.fuel,
                moved=unit.moved,
                is_submerged=unit.is_submerged,
                game_id=game_id,
            )
            unit_items.append((_php_int(len(unit_items)), unit_ser))

    # Loaded (cargo) units: give each a derived id off its carrier so that
    # carrier and cargo never collide. Cargo ids are stable across snapshots
    # as long as the transport is.
    for player_idx, unit_list in sorted(state.units.items()):
        pid = p0_id if player_idx == 0 else p1_id
        for transport in unit_list:
            carrier_id = transport.unit_id if transport.unit_id > 0 else fallback_id
            for slot, cargo in enumerate(transport.loaded_units):
                uid = cargo.unit_id if cargo.unit_id > 0 else (carrier_id * 10_000 + slot + 1)
                unit_ser = _serialize_unit(
                    unit_id=uid,
                    player_id=pid,
                    unit_type=cargo.unit_type,
                    col=transport.pos[1],
                    row=transport.pos[0],
                    hp=cargo.hp,
                    ammo=cargo.ammo,
                    fuel=cargo.fuel,
                    moved=True,
                    is_submerged=False,
                    game_id=game_id,
                )
                unit_items.append((_php_int(len(unit_items)), unit_ser))

    units_array = _php_array(unit_items)

    # capture_win: cap_limit - 2, or 999 for unlimited
    cap_lim = state.map_data.cap_limit
    capture_win = (cap_lim - 2) if cap_lim and cap_lim > 2 else 999

    # ---- Top-level awbwGame object ----
    fields = [
        ("players",          players_array),
        ("buildings",        buildings_array),
        ("units",            units_array),
        ("id",               _php_long(game_id)),
        ("name",             _php_str(game_name)),
        ("password",         _php_null()),
        ("creator",          _php_long(p0_id)),
        ("maps_id",          _php_int(state.map_data.map_id)),
        # AWBW top-level `funds` is the per-property income game setting (not treasury).
        # Per-player current treasuries are serialized in the players array above via state.funds[i].
        ("funds",            _php_int(1000)),
        ("starting_funds",   _php_int(0)),
        ("weather_type",     _php_str(_weather_type_str(state))),
        ("fog",              _php_bool_str(False)),
        ("use_powers",       _php_bool_str(True)),
        ("official",         _php_bool_str(False)),
        ("league",           _php_null()),
        ("team",             _php_bool_str(False)),
        ("turn",             _php_long(active_pid)),
        ("day",              _php_int(state.turn)),
        ("start_date",       _php_str(start_date)),
        ("activity_date",    _php_null()),
        ("end_date",         _php_null()),
        ("weather_code",     _php_str(_weather_code_str(state))),
        ("weather_start",    _php_null()),
        ("win_condition",    _php_null()),
        ("active",           _php_str("Y")),
        ("capture_win",      _php_int(capture_win)),
        ("comment",          _php_null()),
        ("type",             _php_str("N")),
        ("aet_date",         _php_null()),
        ("aet_interval",     _php_int(-1)),
        ("boot_interval",    _php_int(-1)),
        ("max_rating",       _php_null()),
        ("min_rating",       _php_int(0)),
        ("timers_initial",   _php_int(0)),
        ("timers_increment", _php_int(0)),
        ("timers_max_turn",  _php_int(0)),
    ]
    return _php_object("awbwGame", fields)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_awbw_replay(
    snapshots: list[GameState],
    output_path: str | Path,
    game_id: int = 999999,
    game_name: str = "AI vs AI",
    start_date: str = "2026-01-01 00:00:00",
    full_trace: Optional[list[dict]] = None,
) -> Path:
    """
    Serialize a list of turn-start GameState snapshots into an AWBW Replay Player
    compatible .zip file at *output_path*.

    Each snapshot corresponds to the state at the START of one player-turn (i.e.,
    taken after END_TURN processes income/fuel but before the next player acts).

    If `full_trace` is provided (typically `final_state.full_trace`), a second
    gzipped entry `a<game_id>` is appended containing the `p:` per-action stream.
    This unlocks per-move animation in the viewer (ReplayVersion becomes 2).
    In that case the PHP snapshot lines are **not** taken from *snapshots*;
    they are rebuilt in the same trace replay pass as the action stream so
    ``units_id`` values stay aligned (matching ``write_awbw_replay_from_trace``).
    The first element of *snapshots* is still used for map/CO/tier metadata.
    Passing `full_trace=None` keeps the legacy single-entry zip from *snapshots*.

    Returns the resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not snapshots:
        raise ValueError("No snapshots to export.")

    first = snapshots[0]
    if full_trace:
        from tools.export_awbw_replay_actions import (
            _rebuild_and_emit_with_snapshots,
            build_action_stream_text_from_buckets,
            write_action_stream_entry,
        )

        snapshots_for_zip, buckets = _rebuild_and_emit_with_snapshots(
            full_trace=full_trace,
            map_data=first.map_data,
            co0=first.co_states[0].co_id,
            co1=first.co_states[1].co_id,
            tier_name=first.tier_name,
            p0_id=P0_PLAYER_ID,
            p1_id=P1_PLAYER_ID,
        )
    else:
        snapshots_for_zip = snapshots

    # Force visually-distinct OS/BM countries regardless of the map's canonical
    # country_to_player mapping. This guarantees P0 renders red (Orange Star)
    # and P1 renders blue (Blue Moon) in the AWBW Replay Player, and keeps
    # owned-building sprites consistent with unit colors via player_to_country.
    p0_cid = DEFAULT_P0_COUNTRY  # 1 = Orange Star
    p1_cid = DEFAULT_P1_COUNTRY  # 2 = Blue Moon
    player_to_country = {0: p0_cid, 1: p1_cid}

    lines = []
    for snap in snapshots_for_zip:
        line = serialize_turn_snapshot(
            state=snap,
            game_id=game_id,
            player_ids=(P0_PLAYER_ID, P1_PLAYER_ID),
            users_ids=(P0_USERS_ID, P1_USERS_ID),
            countries=(p0_cid, p1_cid),
            player_to_country=player_to_country,
            game_name=game_name,
            start_date=start_date,
        )
        lines.append(line)

    game_state_text = "\n".join(lines)

    # Gzip-compress the game state text
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        gz.write(game_state_text.encode("utf-8"))
    compressed = buf.getvalue()

    # Write a zip with one entry named after the game_id (matches official AWBW format)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(str(game_id), compressed)

    # Optionally append the p: action stream as a second gzipped entry. Kept
    # as a separate step so any failure in action-stream construction does
    # not break the base (game-state-only) zip that the viewer can still
    # consume via ReplayVersion 1.
    if full_trace:
        try:
            action_text = build_action_stream_text_from_buckets(buckets)
            if action_text:
                write_action_stream_entry(output_path, action_text, game_id)
            n_lines = len(action_text.rstrip().split("\n")) if action_text else 0
            log.info(
                "AWBW replay export: game_id=%s php_snapshot_lines=%s p_envelopes=%s "
                "(viewer matches each p: line to TurnData by player id + day)",
                game_id,
                len(snapshots_for_zip),
                n_lines,
            )
        except Exception as exc:
            log.warning(
                "Failed to append p: action stream to %s: %s — "
                "zip is valid but stays ReplayVersion 1 (snapshot-only, no per-move animation).",
                output_path, exc,
            )

    return output_path


# ---------------------------------------------------------------------------
# Reconstruct a zip from a .trace.json record (no live game required)
# ---------------------------------------------------------------------------

def write_awbw_replay_from_trace(
    trace_record: dict,
    output_path: str | Path,
    map_pool_path: str | Path = "data/gl_map_pool.json",
    maps_dir: str | Path = "data/maps",
    game_id: Optional[int] = None,
    game_name: str = "AI vs AI",
    start_date: str = "2026-01-01 00:00:00",
) -> Path:
    """Build an AWBW Replay Player zip entirely from a ``trace_record`` dict.

    The record is the JSON object written by ``tools/export_awbw_replay_actions._save_trace``
    (or the ``*.trace.json`` files in ``replays/``). It must contain at least:

        {
            "map_id":     <int>,
            "co0":        <int>,
            "co1":        <int>,
            "tier":       <str>,   # optional, defaults to "T2"
            "full_trace": [...]    # list of action dicts
        }

    Both the snapshot stream and the ``a<game_id>`` action stream are produced
    in a **single** trace replay pass — they share the same engine ``state``
    object. This guarantees that unit IDs in snapshots match ``units_id``
    values inside Move/Build action JSON, which the C# viewer requires for
    per-action stepping (``MoveAction.SetupAndUpdate`` looks units up by ID;
    a mismatch throws ``ReplayMissingUnitException`` and forces the viewer
    to snap to turn boundaries).
    """
    from tools.export_awbw_replay_actions import (
        _rebuild_and_emit_with_snapshots,
        build_action_stream_text_from_buckets,
        write_action_stream_entry,
    )

    output_path = Path(output_path)
    map_id     = trace_record["map_id"]
    co0        = trace_record["co0"]
    co1        = trace_record["co1"]
    tier       = trace_record.get("tier", "T2")
    full_trace = trace_record["full_trace"]

    if game_id is None:
        game_id = int(output_path.stem) if output_path.stem.isdigit() else 999999

    map_data = load_map(map_id, Path(map_pool_path), Path(maps_dir))

    snapshots, buckets = _rebuild_and_emit_with_snapshots(
        full_trace=full_trace,
        map_data=map_data,
        co0=co0,
        co1=co1,
        tier_name=tier,
        p0_id=P0_PLAYER_ID,
        p1_id=P1_PLAYER_ID,
    )

    action_text = build_action_stream_text_from_buckets(buckets)

    # Write the snapshot zip first (full_trace=None to skip the in-built
    # double-replay action-stream append; we'll inject our pre-built text).
    path = write_awbw_replay(
        snapshots=snapshots,
        output_path=output_path,
        game_id=game_id,
        game_name=game_name,
        start_date=start_date,
        full_trace=None,
    )

    if action_text:
        try:
            write_action_stream_entry(path, action_text, game_id)
        except Exception as exc:
            log.warning(
                "Failed to write p: action stream to %s: %s — "
                "zip is valid but stays ReplayVersion 1 (snapshot-only).",
                path, exc,
            )

    return path
