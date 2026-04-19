"""
AWBW terrain definitions. Terrain IDs → metadata.

Source: https://awbw.amarriner.com/terrain_map.php (authoritative)

ID ranges:
  1:       Plain
  2:       Mountain
  3:       Wood
  4-14:    Rivers (11 orientation variants)
  15-27:   Roads (13 variants, including H/V bridges at 26-27)
  28:      Sea
  29-32:   Shoals (4 orientation variants)
  33:      Reef

  34:      Neutral City
  35:      Neutral Base
  36:      Neutral Airport
  37:      Neutral Port

  38-57:   Orange Star (38-42), Blue Moon (43-47), Green Earth (48-52), Yellow Comet (53-57)
           Each country: city / base / airport / port / HQ
  [58-80 unassigned in AWBW terrain table]
  81-85:   Red Fire    (city / base / airport / port / HQ)
  86-90:   Grey Sky
  91-95:   Black Hole
  96-100:  Brown Desert

  101-110: Pipes (piperunner-only; 10 orientation variants)
  111:     Missile Silo (full — infantry/mech can fire)
  112:     Missile Silo Empty (after firing)
  113-114: Pipe Seams (identical movement properties to Pipe)
  115-116: Broken Pipe Seams (identical to Plains after destruction)

  117-121: Amber Blossom   (airport / base / city / HQ / port)  ← non-sequential order!
  122-126: Jade Sun        (airport / base / city / HQ / port)

  127-137: Comm Towers (capturable, +10% attack bonus per owned tower)
           127=AB, 128=BH, 129=BM, 130=BD, 131=GE, 132=JS,
           133=Neutral, 134=OS, 135=RF, 136=YC, 137=GS

  138-148: Labs (capturable, capture grants unique CO ability)
           138=AB, 139=BH, 140=BM, 141=BD, 142=GE, 143=GS, 144=JS,
           145=Neutral, 146=OS, 147=RF, 148=YC

  149-155: Cobalt Ice      (airport / base / city / comm_tower / HQ / lab / port)
  156-162: Pink Cosmos
  163-169: Teal Galaxy
  170-176: Purple Lightning
  [177-180 unassigned]
  181-187: Acid Rain
  188-194: White Nova
  195:     Teleport Tile (movement cost = 0 for all types)
  196-202: Azure Asteroid
  203-209: Noir Eclipse
  210-216: Silver Claw
  217-223: Umber Wilds

  For countries CI (11) through UW (20): layout is
    airport / base / city / comm_tower / HQ / lab / port  (alphabetical by tile type)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Movement type constants
# ---------------------------------------------------------------------------
MOVE_INF      = "infantry"   # Infantry / Mech on river (boot)
MOVE_MECH     = "mech"       # Mech (better off-road than infantry)
MOVE_TREAD    = "tread"      # Tanks, Md. Tank, Neotank, Megatank
MOVE_TIRE_A   = "tire_a"     # Recon (better off-road tire)
MOVE_TIRE_B   = "tire_b"     # APC, Artillery, Rockets, AA, Missiles (standard tire)
MOVE_AIR      = "air"        # All air units
MOVE_SEA      = "sea"        # Battleship, Carrier, Submarine, Cruiser
MOVE_LANDER   = "lander"     # Lander, Gunboat, Black Boat
MOVE_PIPELINE = "pipe"       # Piperunner only

INF_PASSABLE = 99  # sentinel: tile is impassable for this move type


# ---------------------------------------------------------------------------
# TerrainInfo dataclass
# ---------------------------------------------------------------------------
@dataclass
class TerrainInfo:
    id: int
    name: str
    defense: int               # defense stars (0–4)
    is_property: bool          # can be captured
    is_hq: bool
    is_lab: bool
    is_comm_tower: bool        # grants +10% attack bonus when owned
    is_base: bool              # produces ground units
    is_airport: bool           # produces air units
    is_port: bool              # produces naval units
    country_id: Optional[int]  # None = neutral; AWBW 1-indexed country ID (1=OS, 2=BM, …)
    move_costs: dict[str, int] # move_type → cost (missing key = impassable)


# ---------------------------------------------------------------------------
# Move-cost templates
# ---------------------------------------------------------------------------
def _plain_costs() -> dict[str, int]:
    return {MOVE_INF: 1, MOVE_MECH: 1, MOVE_TREAD: 1,
            MOVE_TIRE_A: 2, MOVE_TIRE_B: 2, MOVE_AIR: 1}

def _mountain_costs() -> dict[str, int]:
    return {MOVE_INF: 2, MOVE_MECH: 1, MOVE_AIR: 1}

def _wood_costs() -> dict[str, int]:
    return {MOVE_INF: 1, MOVE_MECH: 1, MOVE_TREAD: 3,
            MOVE_TIRE_A: 3, MOVE_TIRE_B: 3, MOVE_AIR: 1}

def _road_costs() -> dict[str, int]:
    return {MOVE_INF: 1, MOVE_MECH: 1, MOVE_TREAD: 1,
            MOVE_TIRE_A: 1, MOVE_TIRE_B: 1, MOVE_AIR: 1}

def _river_costs() -> dict[str, int]:
    # Rivers: foot (infantry) cost 2, boot (mech) cost 1, air cost 1; all others impassable
    return {MOVE_INF: 2, MOVE_MECH: 1, MOVE_AIR: 1}

def _sea_costs() -> dict[str, int]:
    return {MOVE_SEA: 1, MOVE_LANDER: 1, MOVE_AIR: 1}

def _shoal_costs() -> dict[str, int]:
    # All ground types + lander; sea units cannot enter shoals
    return {MOVE_INF: 1, MOVE_MECH: 1, MOVE_TREAD: 1,
            MOVE_TIRE_A: 1, MOVE_TIRE_B: 1, MOVE_AIR: 1, MOVE_LANDER: 1}

def _reef_costs() -> dict[str, int]:
    return {MOVE_SEA: 2, MOVE_LANDER: 2, MOVE_AIR: 1}

def _property_costs() -> dict[str, int]:
    return {MOVE_INF: 1, MOVE_MECH: 1, MOVE_TREAD: 1,
            MOVE_TIRE_A: 1, MOVE_TIRE_B: 1, MOVE_AIR: 1}

def _pipe_costs() -> dict[str, int]:
    return {MOVE_PIPELINE: 1}

def _impassable_costs() -> dict[str, int]:
    return {}

def _silo_costs() -> dict[str, int]:
    # Missile silos: ground units + air can enter; sea/lander cannot
    return {MOVE_INF: 1, MOVE_MECH: 1, MOVE_TREAD: 1,
            MOVE_TIRE_A: 1, MOVE_TIRE_B: 1, MOVE_AIR: 1}

def _teleport_costs() -> dict[str, int]:
    # Teleport tiles: cost 0 for ALL movement types
    return {MOVE_INF: 0, MOVE_MECH: 0, MOVE_TREAD: 0,
            MOVE_TIRE_A: 0, MOVE_TIRE_B: 0, MOVE_AIR: 0,
            MOVE_SEA: 0, MOVE_LANDER: 0, MOVE_PIPELINE: 0}


# ---------------------------------------------------------------------------
# Country name lookup (AWBW 1-indexed country IDs)
# ---------------------------------------------------------------------------
_COUNTRY_NAMES: dict[int, str] = {
    1:  "Orange Star",
    2:  "Blue Moon",
    3:  "Green Earth",
    4:  "Yellow Comet",
    5:  "Black Hole",
    6:  "Red Fire",
    7:  "Grey Sky",
    8:  "Brown Desert",
    9:  "Amber Blossom",
    10: "Jade Sun",
    11: "Cobalt Ice",
    12: "Pink Cosmos",
    13: "Teal Galaxy",
    14: "Purple Lightning",
    15: "Acid Rain",
    16: "White Nova",
    17: "Azure Asteroid",
    18: "Noir Eclipse",
    19: "Silver Claw",
    20: "Umber Wilds",
}


# ---------------------------------------------------------------------------
# Helpers to construct TerrainInfo entries
# ---------------------------------------------------------------------------
def _T(
    tid: int, name: str, defense: int,
    is_property: bool = False,
    is_hq: bool = False,
    is_lab: bool = False,
    is_comm_tower: bool = False,
    is_base: bool = False,
    is_airport: bool = False,
    is_port: bool = False,
    country_id: Optional[int] = None,
    move_costs: Optional[dict] = None,
) -> TerrainInfo:
    if move_costs is None:
        move_costs = _property_costs() if is_property else _plain_costs()
    return TerrainInfo(
        id=tid, name=name, defense=defense,
        is_property=is_property, is_hq=is_hq, is_lab=is_lab,
        is_comm_tower=is_comm_tower, is_base=is_base,
        is_airport=is_airport, is_port=is_port,
        country_id=country_id, move_costs=move_costs,
    )

def _prop(tid: int, label: str, country_id: Optional[int] = None,
          is_hq: bool = False, is_lab: bool = False, is_comm_tower: bool = False,
          is_base: bool = False, is_airport: bool = False, is_port: bool = False,
          defense: int = 3) -> TerrainInfo:
    """Construct a standard capturable property."""
    cname = _COUNTRY_NAMES.get(country_id, "Unknown") if country_id else "neutral"
    return _T(tid, f"{label} ({cname})", defense,
              is_property=True, is_hq=is_hq, is_lab=is_lab,
              is_comm_tower=is_comm_tower, is_base=is_base,
              is_airport=is_airport, is_port=is_port,
              country_id=country_id, move_costs=_property_costs())


# ---------------------------------------------------------------------------
# Build TERRAIN_TABLE
# ---------------------------------------------------------------------------
def _build_table() -> dict[int, TerrainInfo]:
    table: dict[int, TerrainInfo] = {}

    # ---- 1–3: Basic natural terrain ----
    table[1]  = _T(1,  "Plain",    1, move_costs=_plain_costs())
    table[2]  = _T(2,  "Mountain", 4, move_costs=_mountain_costs())
    table[3]  = _T(3,  "Wood",     2, move_costs=_wood_costs())

    # ---- 4–14: Rivers (11 orientation variants) ----
    _RIVER_NAMES = {
        4: "HRiver", 5: "VRiver", 6: "CRiver",
        7: "ESRiver", 8: "SWRiver", 9: "WNRiver", 10: "NERiver",
        11: "ESWRiver", 12: "SWNRiver", 13: "WNERiver", 14: "NESRiver",
    }
    for tid in range(4, 15):
        table[tid] = _T(tid, _RIVER_NAMES[tid], 0, move_costs=_river_costs())

    # ---- 15–27: Roads and Bridges ----
    _ROAD_NAMES = {
        15: "HRoad", 16: "VRoad", 17: "CRoad",
        18: "ESRoad", 19: "SWRoad", 20: "WNRoad", 21: "NERoad",
        22: "ESWRoad", 23: "SWNRoad", 24: "WNERoad", 25: "NESRoad",
        26: "HBridge", 27: "VBridge",
    }
    for tid in range(15, 28):
        table[tid] = _T(tid, _ROAD_NAMES[tid], 0, move_costs=_road_costs())

    # ---- 28: Sea ----
    table[28] = _T(28, "Sea", 0, move_costs=_sea_costs())

    # ---- 29–32: Shoals ----
    _SHOAL_NAMES = {29: "HShoal", 30: "HShoalN", 31: "VShoal", 32: "VShoalE"}
    for tid in range(29, 33):
        table[tid] = _T(tid, _SHOAL_NAMES[tid], 0, move_costs=_shoal_costs())

    # ---- 33: Reef ----
    table[33] = _T(33, "Reef", 1, move_costs=_reef_costs())

    # ---- 34–37: Neutral properties ----
    table[34] = _prop(34, "City",    defense=3)
    table[35] = _prop(35, "Base",    defense=3, is_base=True)
    table[36] = _prop(36, "Airport", defense=3, is_airport=True)
    table[37] = _prop(37, "Port",    defense=3, is_port=True)

    # ---- 38–57: Orange Star, Blue Moon, Green Earth, Yellow Comet ----
    # Each country: city / base / airport / port / HQ
    _EARLY_COUNTRIES = [
        (1,  38,  "Orange Star"),
        (2,  43,  "Blue Moon"),
        (3,  48,  "Green Earth"),
        (4,  53,  "Yellow Comet"),
    ]
    for cid, base_id, _name in _EARLY_COUNTRIES:
        table[base_id + 0] = _prop(base_id + 0, "City",    cid)
        table[base_id + 1] = _prop(base_id + 1, "Base",    cid, is_base=True)
        table[base_id + 2] = _prop(base_id + 2, "Airport", cid, is_airport=True)
        table[base_id + 3] = _prop(base_id + 3, "Port",    cid, is_port=True)
        table[base_id + 4] = _prop(base_id + 4, "HQ",      cid, is_hq=True, defense=4)

    # IDs 58–80 are unassigned in the AWBW terrain table (no country uses them)

    # ---- 81–100: Red Fire, Grey Sky, Black Hole, Brown Desert ----
    _MID_COUNTRIES = [
        (6,  81,  "Red Fire"),
        (7,  86,  "Grey Sky"),
        (5,  91,  "Black Hole"),
        (8,  96,  "Brown Desert"),
    ]
    for cid, base_id, _name in _MID_COUNTRIES:
        table[base_id + 0] = _prop(base_id + 0, "City",    cid)
        table[base_id + 1] = _prop(base_id + 1, "Base",    cid, is_base=True)
        table[base_id + 2] = _prop(base_id + 2, "Airport", cid, is_airport=True)
        table[base_id + 3] = _prop(base_id + 3, "Port",    cid, is_port=True)
        table[base_id + 4] = _prop(base_id + 4, "HQ",      cid, is_hq=True, defense=4)

    # ---- 101–116: Pipes, Missile Silos, Pipe Seams, Broken Seams ----
    _PIPE_NAMES = {
        101: "VPipe", 102: "HPipe",
        103: "NEPipe", 104: "ESPipe", 105: "SWPipe", 106: "WNPipe",
        107: "NPipeEnd", 108: "EPipeEnd", 109: "SPipeEnd", 110: "WPipeEnd",
    }
    for tid in range(101, 111):
        table[tid] = _T(tid, _PIPE_NAMES[tid], 0, move_costs=_pipe_costs())

    table[111] = _T(111, "Missile Silo",       3, move_costs=_silo_costs())
    table[112] = _T(112, "Missile Silo Empty", 3, move_costs=_silo_costs())

    # Pipe Seams: same movement as pipes (impassable except piperunner)
    table[113] = _T(113, "HPipe Seam", 0, move_costs=_pipe_costs())
    table[114] = _T(114, "VPipe Seam", 0, move_costs=_pipe_costs())

    # Broken Pipe Seams: same as plains after destruction
    table[115] = _T(115, "HPipe Rubble", 1, move_costs=_plain_costs())
    table[116] = _T(116, "VPipe Rubble", 1, move_costs=_plain_costs())

    # ---- 117–126: Amber Blossom and Jade Sun ----
    # Note: these countries use non-sequential ordering (airport/base/city/HQ/port)
    # AB
    table[117] = _prop(117, "Airport", 9,  is_airport=True)
    table[118] = _prop(118, "Base",    9,  is_base=True)
    table[119] = _prop(119, "City",    9)
    table[120] = _prop(120, "HQ",      9,  is_hq=True, defense=4)
    table[121] = _prop(121, "Port",    9,  is_port=True)
    # JS
    table[122] = _prop(122, "Airport", 10, is_airport=True)
    table[123] = _prop(123, "Base",    10, is_base=True)
    table[124] = _prop(124, "City",    10)
    table[125] = _prop(125, "HQ",      10, is_hq=True, defense=4)
    table[126] = _prop(126, "Port",    10, is_port=True)

    # ---- 127–137: Comm Towers ----
    _COMM_TOWER_COUNTRIES = {
        127: 9,    # AB
        128: 5,    # BH
        129: 2,    # BM
        130: 8,    # BD
        131: 3,    # GE
        132: 10,   # JS
        133: None, # Neutral
        134: 1,    # OS
        135: 6,    # RF
        136: 4,    # YC
        137: 7,    # GS
    }
    for tid, cid in _COMM_TOWER_COUNTRIES.items():
        table[tid] = _prop(tid, "Com Tower", cid, is_comm_tower=True)

    # ---- 138–148: Labs ----
    _LAB_COUNTRIES = {
        138: 9,    # AB
        139: 5,    # BH
        140: 2,    # BM
        141: 8,    # BD
        142: 3,    # GE
        143: 7,    # GS
        144: 10,   # JS
        145: None, # Neutral
        146: 1,    # OS
        147: 6,    # RF
        148: 4,    # YC
    }
    for tid, cid in _LAB_COUNTRIES.items():
        table[tid] = _prop(tid, "Lab", cid, is_lab=True)

    # ---- 149–223: Extended countries (CI through UW) ----
    # Each country has 7 properties: airport / base / city / comm_tower / HQ / lab / port
    _EXT_COUNTRIES = [
        (11, 149),  # Cobalt Ice
        (12, 156),  # Pink Cosmos
        (13, 163),  # Teal Galaxy
        (14, 170),  # Purple Lightning
        # 177-180 unassigned
        (15, 181),  # Acid Rain
        (16, 188),  # White Nova
        # 195 = Teleporter
        (17, 196),  # Azure Asteroid
        (18, 203),  # Noir Eclipse
        (19, 210),  # Silver Claw
        (20, 217),  # Umber Wilds
    ]
    for cid, base_id in _EXT_COUNTRIES:
        # Layout: airport(+0) / base(+1) / city(+2) / comm_tower(+3) / hq(+4) / lab(+5) / port(+6)
        table[base_id + 0] = _prop(base_id + 0, "Airport",   cid, is_airport=True)
        table[base_id + 1] = _prop(base_id + 1, "Base",      cid, is_base=True)
        table[base_id + 2] = _prop(base_id + 2, "City",      cid)
        table[base_id + 3] = _prop(base_id + 3, "Com Tower", cid, is_comm_tower=True)
        table[base_id + 4] = _prop(base_id + 4, "HQ",        cid, is_hq=True, defense=4)
        table[base_id + 5] = _prop(base_id + 5, "Lab",       cid, is_lab=True)
        table[base_id + 6] = _prop(base_id + 6, "Port",      cid, is_port=True)

    # ---- 195: Teleport Tile ----
    table[195] = _T(195, "Teleporter", 0, move_costs=_teleport_costs())

    return table


TERRAIN_TABLE: dict[int, TerrainInfo] = _build_table()

# Fallback for unknown IDs: treat as impassable (so bugs surface quickly)
_FALLBACK = TerrainInfo(
    id=-1, name="Unknown", defense=0,
    is_property=False, is_hq=False, is_lab=False, is_comm_tower=False,
    is_base=False, is_airport=False, is_port=False,
    country_id=None,
    move_costs=_plain_costs(),  # plain-passable to avoid hard crashes; log warnings
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_terrain(terrain_id: int) -> TerrainInfo:
    """Return TerrainInfo for the given ID; falls back to plain-like unknown for unmapped IDs."""
    t = TERRAIN_TABLE.get(terrain_id)
    if t is not None:
        return t
    # No dynamic guessing — the full table is explicit. Unknown IDs are genuinely unknown.
    return _FALLBACK


def is_hq(terrain_id: int) -> bool:
    t = TERRAIN_TABLE.get(terrain_id)
    return t is not None and t.is_hq


def is_lab(terrain_id: int) -> bool:
    t = TERRAIN_TABLE.get(terrain_id)
    return t is not None and t.is_lab


def is_comm_tower(terrain_id: int) -> bool:
    t = TERRAIN_TABLE.get(terrain_id)
    return t is not None and t.is_comm_tower


def is_property(terrain_id: int) -> bool:
    t = TERRAIN_TABLE.get(terrain_id)
    return t is not None and t.is_property


def get_country(terrain_id: int) -> Optional[int]:
    t = TERRAIN_TABLE.get(terrain_id)
    return t.country_id if t is not None else None


def country_id_for_player_seat(country_to_player: dict[int, int], player: int) -> Optional[int]:
    """Inverse of map load seating: which AWBW ``country_id`` sits on engine ``player``."""
    for cid, p in country_to_player.items():
        if p == player:
            return cid
    return None


def property_building_signature(t: TerrainInfo) -> tuple[bool, bool, bool, bool, bool, bool]:
    return (t.is_hq, t.is_lab, t.is_comm_tower, t.is_base, t.is_airport, t.is_port)


def property_terrain_id_for_country_and_kind(
    country_id: int,
    *,
    is_hq: bool,
    is_lab: bool,
    is_comm_tower: bool,
    is_base: bool,
    is_airport: bool,
    is_port: bool,
) -> Optional[int]:
    """
    Find the terrain ID for a capturable property of this ``country_id`` and building kind.

    Used after full capture so ``map_data.terrain`` matches faction-coloured AWBW tiles.
    """
    sig = (is_hq, is_lab, is_comm_tower, is_base, is_airport, is_port)
    for tid, t in TERRAIN_TABLE.items():
        if not t.is_property or t.country_id != country_id:
            continue
        if property_building_signature(t) == sig:
            return int(tid)
    return None


def property_terrain_id_after_owner_change(
    old_terrain_id: int,
    new_owner_player: int,
    country_to_player: dict[int, int],
) -> Optional[int]:
    """New terrain tile ID when ``new_owner_player`` fully owns this property tile."""
    old = get_terrain(old_terrain_id)
    if not old.is_property:
        return None
    cid = country_id_for_player_seat(country_to_player, new_owner_player)
    if cid is None:
        return None
    return property_terrain_id_for_country_and_kind(
        cid,
        is_hq=old.is_hq,
        is_lab=old.is_lab,
        is_comm_tower=old.is_comm_tower,
        is_base=old.is_base,
        is_airport=old.is_airport,
        is_port=old.is_port,
    )


def get_move_cost(terrain_id: int, move_type: str) -> int:
    t = TERRAIN_TABLE.get(terrain_id, _FALLBACK)
    return t.move_costs.get(move_type, INF_PASSABLE)
