"""
Parse AWBW CSV terrain files + gl_map_pool.json metadata into MapData.

CSV format: one row per line, comma-separated terrain IDs, no header.

Engine seats (used everywhere: training, play UI, encoder):
  **Player 0 = red seat** — the side you train as / human controls in ``/play/``.
  **Player 1 = blue seat** — opponent. On symmetric starts with units on both sides,
  **P0 moves first** (see ``make_initial_state`` opening rule).

Country-to-player assignment (default):
  The first country encountered while scanning the grid (row-major, top-left → bottom-right)
  becomes player 0; the second becomes player 1. This follows the visual convention that
  the "top" or "left" side of symmetric maps belongs to P0.

Optional override: map pool entry ``p0_country_id`` (AWBW ``country_id`` from ``terrain``)
  forces that country onto **player 0** and the other competitive country onto **player 1**,
  and remaps predeployed ``player`` indices accordingly. Use when AWBW lobby / art expects
  a specific faction in the red seat. **Changing seating invalidates old checkpoints** for
  that map — retrain after toggling.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.terrain import get_terrain, is_hq, is_lab, is_property, get_country
from engine.predeployed import PredeployedUnitSpec, load_predeployed_units_file


def apply_p0_country_id_seating(
    properties: list[PropertyState],
    scan_country_to_player: dict[int, int],
    p0_country_id: int,
    predeployed: list[PredeployedUnitSpec],
    *,
    map_id: int,
    map_name: str,
) -> tuple[dict[int, int], list[PredeployedUnitSpec], dict[int, list], dict[int, list]]:
    """
    Remap property owners, HQ/lab index lists, and predeployed unit ``player`` so that
    ``p0_country_id`` (AWBW terrain ``country_id``) sits on **engine player 0** (red /
    first seat) and the other map country on **player 1** (blue / second seat).

    ``scan_country_to_player`` is the row-major scan result before this remap.
    """
    if len(scan_country_to_player) != 2:
        raise ValueError(
            f"Map {map_id} ({map_name!r}): p0_country_id requires exactly two distinct "
            f"player countries; got {list(scan_country_to_player.keys())}"
        )
    if p0_country_id not in scan_country_to_player:
        raise ValueError(
            f"Map {map_id} ({map_name!r}): p0_country_id={p0_country_id} is not one of the "
            f"map countries {list(scan_country_to_player.keys())}"
        )
    other_c = next(c for c in scan_country_to_player if c != p0_country_id)
    new_ctp = {p0_country_id: 0, other_c: 1}
    old_to_new: dict[int, int] = {
        scan_country_to_player[c]: new_ctp[c] for c in scan_country_to_player
    }

    for prop in properties:
        cid = get_country(prop.terrain_id)
        if cid is not None and cid in new_ctp:
            prop.owner = new_ctp[cid]

    hq_positions: dict[int, list] = {0: [], 1: []}
    lab_positions: dict[int, list] = {0: [], 1: []}
    for prop in properties:
        if prop.owner is None:
            continue
        if prop.is_hq:
            hq_positions[prop.owner].append((prop.row, prop.col))
        if prop.is_lab:
            lab_positions[prop.owner].append((prop.row, prop.col))

    new_specs: list[PredeployedUnitSpec] = [
        PredeployedUnitSpec(
            row=s.row,
            col=s.col,
            player=old_to_new[s.player],
            unit_type=s.unit_type,
            hp=s.hp,
        )
        for s in predeployed
    ]
    return new_ctp, new_specs, hq_positions, lab_positions


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PropertyState:
    terrain_id: int
    row: int
    col: int
    owner: Optional[int]    # 0, 1, or None (neutral)
    capture_points: int     # 20 = fully owned; <20 = in progress of being captured
    is_hq: bool
    is_lab: bool
    is_comm_tower: bool
    is_base: bool
    is_airport: bool
    is_port: bool

    def __repr__(self) -> str:
        kind = "HQ" if self.is_hq else ("Lab" if self.is_lab else ("Tower" if self.is_comm_tower else "Prop"))
        return f"{kind}(tid={self.terrain_id}, pos=({self.row},{self.col}), owner={self.owner})"


@dataclass
class TierInfo:
    tier_name: str
    enabled: bool
    co_ids: list[int]
    co_names: list[str]


@dataclass
class MapData:
    map_id: int
    name: str
    map_type: str               # e.g. "std"
    terrain: list[list[int]]    # [row][col] → terrain ID
    height: int
    width: int
    cap_limit: int
    unit_limit: int
    unit_bans: list[str]
    tiers: list[TierInfo]
    objective_type: Optional[str]   # "hq" | "lab" | None (unit wipe / cap limit only)
    properties: list[PropertyState]
    hq_positions: dict[int, list[tuple[int, int]]]   # player → [(row, col), ...]
    lab_positions: dict[int, list[tuple[int, int]]]  # player → [(row, col), ...]
    country_to_player: dict[int, int]                # country_id → 0 red/first or 1 blue/second
    # Optional `data/maps/<map_id>_units.json` — AWBW csv export is terrain-only
    predeployed_specs: list[PredeployedUnitSpec] = field(default_factory=list)

    def get_enabled_tiers(self) -> list[TierInfo]:
        return [t for t in self.tiers if t.enabled]

    def get_tier(self, tier_name: str) -> Optional[TierInfo]:
        for t in self.tiers:
            if t.tier_name == tier_name:
                return t
        return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_map(
    map_id: int,
    map_pool_path: Path,
    maps_dir: Path,
) -> MapData:
    """
    Load a map by ID.

    Parameters
    ----------
    map_id:         AWBW numeric map ID.
    map_pool_path:  Path to gl_map_pool.json.
    maps_dir:       Directory containing <map_id>.csv files.

    Raises
    ------
    ValueError      if map_id is not in the pool.
    FileNotFoundError if the CSV file doesn't exist.
    """
    # ---- Pool metadata ----
    with open(map_pool_path, encoding="utf-8") as f:
        pool: list[dict] = json.load(f)

    meta = next((m for m in pool if m["map_id"] == map_id), None)
    if meta is None:
        available = [m["map_id"] for m in pool]
        raise ValueError(
            f"Map {map_id} not found in pool ({map_pool_path}). "
            f"Available IDs: {available}"
        )

    tiers = [
        TierInfo(
            tier_name=t["tier_name"],
            enabled=t["enabled"],
            co_ids=t["co_ids"],
            co_names=t["co_names"],
        )
        for t in meta.get("tiers", [])
    ]

    # ---- CSV terrain grid ----
    csv_path = maps_dir / f"{map_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Map CSV not found: {csv_path}. "
            "Ensure the maps directory is populated."
        )

    terrain: list[list[int]] = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                terrain.append([int(x) for x in line.split(",")])

    height = len(terrain)
    width  = len(terrain[0]) if terrain else 0

    # ---- Scan properties, assign countries to players ----
    properties: list[PropertyState]       = []
    hq_positions:  dict[int, list]        = {0: [], 1: []}
    lab_positions: dict[int, list]        = {0: [], 1: []}
    country_to_player: dict[int, int]     = {}

    for r, row in enumerate(terrain):
        for c, tid in enumerate(row):
            info = get_terrain(tid)
            if not info.is_property:
                continue

            country = get_country(tid)
            player: Optional[int] = None

            if country is not None:
                if country not in country_to_player:
                    # Assign next available player index (0 then 1).
                    # Comm towers and labs for extended countries carry a country_id —
                    # only assign a player slot for countries that have HQs/production
                    # buildings, not exclusively comm-tower/lab-only appearances.
                    idx = len(country_to_player)
                    if idx < 2:
                        country_to_player[country] = idx
                player = country_to_player.get(country)

            prop = PropertyState(
                terrain_id=tid,
                row=r,
                col=c,
                owner=player,
                capture_points=20,
                is_hq=info.is_hq,
                is_lab=info.is_lab,
                is_comm_tower=info.is_comm_tower,
                is_base=info.is_base,
                is_airport=info.is_airport,
                is_port=info.is_port,
            )
            properties.append(prop)

            if info.is_hq and player is not None and player in hq_positions:
                hq_positions[player].append((r, c))
            if info.is_lab and player is not None and player in lab_positions:
                lab_positions[player].append((r, c))

    # ---- Sanity check: competitive maps must have exactly 2 owned countries ----
    if len(country_to_player) != 2:
        warnings.warn(
            f"Map {map_id} ({meta['name']!r}) has {len(country_to_player)} owned countries "
            f"(expected 2): {list(country_to_player.keys())}. "
            "Player assignment may be incorrect.",
            stacklevel=2,
        )

    predeployed = load_predeployed_units_file(maps_dir / f"{map_id}_units.json")
    for s in predeployed:
        if not (0 <= s.row < height and 0 <= s.col < width):
            raise ValueError(
                f"Predeployed unit at ({s.row},{s.col}) out of bounds for map {map_id} "
                f"({height}x{width})"
            )

    raw_p0 = meta.get("p0_country_id")
    if raw_p0 is not None:
        scan_ctp = dict(country_to_player)
        country_to_player, predeployed, hq_positions, lab_positions = apply_p0_country_id_seating(
            properties,
            scan_ctp,
            int(raw_p0),
            predeployed,
            map_id=map_id,
            map_name=str(meta.get("name", "")),
        )

    # ---- Determine win-condition objective (after optional seating remap) ----
    has_hqs  = any(hq_positions[p]  for p in hq_positions)
    has_labs = any(lab_positions[p] for p in lab_positions)

    if has_hqs:
        objective_type: Optional[str] = "hq"
    elif has_labs:
        objective_type = "lab"
    else:
        objective_type = None

    return MapData(
        map_id=map_id,
        name=meta["name"],
        map_type=meta.get("type", "std"),
        terrain=terrain,
        height=height,
        width=width,
        cap_limit=meta["cap_limit"],
        unit_limit=meta["unit_limit"],
        unit_bans=meta.get("unit_bans", []),
        tiers=tiers,
        objective_type=objective_type,
        properties=properties,
        hq_positions=hq_positions,
        lab_positions=lab_positions,
        country_to_player=country_to_player,
        predeployed_specs=predeployed,
    )


# ---------------------------------------------------------------------------
# Convenience: load all maps in the pool
# ---------------------------------------------------------------------------

def load_all_maps(
    map_pool_path: Path,
    maps_dir: Path,
    *,
    skip_missing: bool = True,
) -> dict[int, MapData]:
    """
    Load every map referenced in gl_map_pool.json.

    Parameters
    ----------
    skip_missing:   If True, silently skip maps whose CSV doesn't exist.
                    If False, raise FileNotFoundError on the first missing CSV.
    """
    with open(map_pool_path, encoding="utf-8") as f:
        pool: list[dict] = json.load(f)

    result: dict[int, MapData] = {}
    for entry in pool:
        mid = entry["map_id"]
        try:
            result[mid] = load_map(mid, map_pool_path, maps_dir)
        except FileNotFoundError:
            if skip_missing:
                continue
            raise

    return result
