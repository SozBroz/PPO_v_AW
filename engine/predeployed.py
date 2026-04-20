"""
Predeployed units: AWBW's text/csv map export is terrain-only. Optional sidecar
`data/maps/<map_id>_units.json` lists starting units so `make_initial_state`
matches AWBW when that file is present.

Schema (JSON object):
  schema_version: int (optional, default 1)
  units: list of {
    "row": int,
    "col": int,
    "player": 0 | 1,
    "unit_type": str,   # UnitType enum name, e.g. "INFANTRY"
    "hp": int (optional, default 100),
    "force_engine_player": 0 | 1 (optional) — after ``p0_country_id`` remap, use this
        engine seat instead of inferring from terrain country under the unit (rare;
        e.g. map 69201 predeploy on another country's base).
  }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from engine.unit import Unit, UnitType, UNIT_STATS


@dataclass(frozen=True)
class PredeployedUnitSpec:
    row: int
    col: int
    player: int
    unit_type: UnitType
    hp: int = 100
    force_engine_player: Optional[int] = None


def _parse_unit_type(name: str) -> UnitType:
    key = name.strip().upper()
    try:
        return UnitType[key]
    except KeyError as e:
        valid = ", ".join(sorted(t.name for t in UnitType))
        raise ValueError(f"Unknown unit_type {name!r}. Expected one of: {valid}") from e


def load_predeployed_units_file(path: Path) -> list[PredeployedUnitSpec]:
    """Load and validate predeployed unit specs from JSON."""
    if not path.exists():
        return []
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw_units = data.get("units")
    if raw_units is None:
        return []
    if not isinstance(raw_units, list):
        raise ValueError(f"{path}: 'units' must be a list")

    out: list[PredeployedUnitSpec] = []
    seen: set[tuple[int, int]] = set()
    for i, u in enumerate(raw_units):
        if not isinstance(u, dict):
            raise ValueError(f"{path}: units[{i}] must be an object")
        try:
            row = int(u["row"])
            col = int(u["col"])
            player = int(u["player"])
            ut = _parse_unit_type(str(u["unit_type"]))
        except KeyError as e:
            raise ValueError(f"{path}: units[{i}] missing field: {e}") from e

        if player not in (0, 1):
            raise ValueError(f"{path}: units[{i}].player must be 0 or 1, got {player}")
        hp = int(u.get("hp", 100))
        if not 1 <= hp <= 100:
            raise ValueError(f"{path}: units[{i}].hp must be 1–100, got {hp}")

        raw_fe = u.get("force_engine_player")
        if raw_fe is None:
            fe: Optional[int] = None
        else:
            fe = int(raw_fe)
            if fe not in (0, 1):
                raise ValueError(
                    f"{path}: units[{i}].force_engine_player must be 0 or 1, got {fe}"
                )

        pos = (row, col)
        if pos in seen:
            raise ValueError(f"{path}: duplicate unit at {pos}")
        seen.add(pos)
        out.append(
            PredeployedUnitSpec(
                row=row, col=col, player=player, unit_type=ut, hp=hp, force_engine_player=fe
            )
        )

    return out


def specs_to_initial_units(specs: list[PredeployedUnitSpec]) -> dict[int, list[Unit]]:
    """Group specs into GameState.units shape."""
    result: dict[int, list[Unit]] = {0: [], 1: []}
    for s in specs:
        stats = UNIT_STATS[s.unit_type]
        ammo = stats.max_ammo if stats.max_ammo > 0 else 0
        u = Unit(
            unit_type=s.unit_type,
            player=s.player,
            hp=s.hp,
            ammo=ammo,
            fuel=stats.max_fuel,
            pos=(s.row, s.col),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
        )
        result[s.player].append(u)
    return result
