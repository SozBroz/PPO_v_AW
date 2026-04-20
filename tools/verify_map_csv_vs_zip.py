#!/usr/bin/env python3
"""
Compare ``data/maps/<maps_id>.csv`` terrain against AWBW's PHP snapshot inside a replay zip.

The first gzipped line of the zip's primary member is parsed with
``tools.diff_replay_zips.load_replay`` (same contract as the AWBW Replay Player).
Each ``awbwBuilding`` entry carries authoritative ``terrain_id`` at ``(y, x)``
(row, col). Any cell listed in the snapshot whose id differs from the CSV is
reported — the usual root cause for ``Illegal move … terrain id=… is not reachable``
when the repo map is stale vs the live site export.

Examples::

  python tools/verify_map_csv_vs_zip.py replays/amarriner_gl/1609533.zip
  python tools/verify_map_csv_vs_zip.py replays/amarriner_gl/1629178.zip --maps-dir data/maps
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.terrain import (
    MOVE_AIR,
    MOVE_INF,
    MOVE_LANDER,
    MOVE_MECH,
    MOVE_PIPELINE,
    MOVE_SEA,
    MOVE_TIRE_A,
    MOVE_TIRE_B,
    MOVE_TREAD,
    get_move_cost,
)

from tools.diff_replay_zips import load_replay

_MOVE_TYPES_ORDERED: tuple[str, ...] = (
    MOVE_INF,
    MOVE_MECH,
    MOVE_TREAD,
    MOVE_TIRE_A,
    MOVE_TIRE_B,
    MOVE_AIR,
    MOVE_SEA,
    MOVE_LANDER,
    MOVE_PIPELINE,
)


def _movement_signature(tid: int) -> tuple[int, ...]:
    """Per-tile costs for every engine move type — catches real CSV drift vs cosmetic country art."""
    return tuple(get_move_cost(int(tid), mt) for mt in _MOVE_TYPES_ORDERED)


def _load_csv_terrain(csv_path: Path) -> list[list[int]]:
    terrain: list[list[int]] = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                terrain.append([int(x) for x in line.split(",")])
    return terrain


def _iter_awbw_building_tiles(frame0: dict[str, Any]) -> Iterator[tuple[int, int, int]]:
    buildings = frame0.get("buildings") or {}
    for _k, b in buildings.items():
        if not isinstance(b, dict):
            continue
        if b.get("__class__") and str(b.get("__class__")) != "awbwBuilding":
            continue
        try:
            row = int(b["y"])
            col = int(b["x"])
            tid = int(b["terrain_id"])
        except (KeyError, TypeError, ValueError):
            continue
        yield row, col, tid


@dataclass
class TerrainMismatch:
    row: int
    col: int
    csv_tid: Optional[int]
    php_tid: int


def diff_csv_vs_zip_frame0(
    *,
    csv_terrain: list[list[int]],
    frame0: dict[str, Any],
    strict_ids: bool = False,
) -> list[TerrainMismatch]:
    h = len(csv_terrain)
    w = len(csv_terrain[0]) if h else 0
    out: list[TerrainMismatch] = []
    for row, col, php_tid in _iter_awbw_building_tiles(frame0):
        if not (0 <= row < h and 0 <= col < w):
            out.append(TerrainMismatch(row, col, None, php_tid))
            continue
        csv_tid = int(csv_terrain[row][col])
        if csv_tid == php_tid:
            continue
        if not strict_ids and _movement_signature(csv_tid) == _movement_signature(php_tid):
            continue
        out.append(TerrainMismatch(row, col, csv_tid, php_tid))
    return out


def verify_zip(
    zip_path: Path,
    *,
    maps_dir: Path,
    map_id_override: int | None = None,
    strict_ids: bool = False,
) -> tuple[int, list[TerrainMismatch]]:
    frames = load_replay(zip_path)
    if not frames:
        raise ValueError(f"empty replay: {zip_path}")
    frame0 = frames[0]
    mid = map_id_override
    if mid is None:
        raw = frame0.get("maps_id")
        if raw is None:
            raise ValueError("frame0 missing maps_id (cannot locate CSV)")
        mid = int(raw)
    csv_path = maps_dir / f"{mid}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing map CSV: {csv_path}")
    csv_terrain = _load_csv_terrain(csv_path)
    mism = diff_csv_vs_zip_frame0(
        csv_terrain=csv_terrain, frame0=frame0, strict_ids=strict_ids
    )
    return mid, mism


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip", type=Path, help="AWBW replay .zip (site or export)")
    ap.add_argument(
        "--maps-dir",
        type=Path,
        default=ROOT / "data" / "maps",
        help="Directory with <map_id>.csv (default: repo data/maps)",
    )
    ap.add_argument(
        "--map-id",
        type=int,
        default=None,
        help="Override maps_id from the zip (rare)",
    )
    ap.add_argument(
        "--strict-ids",
        action="store_true",
        help="Flag any terrain_id inequality (ignore OS/BM recolors with identical move costs)",
    )
    args = ap.parse_args()
    mid, mism = verify_zip(
        args.zip,
        maps_dir=args.maps_dir,
        map_id_override=args.map_id,
        strict_ids=bool(args.strict_ids),
    )
    if not mism:
        print(f"ok maps_id={mid} csv matches PHP buildings in {args.zip.name}")
        return 0
    print(f"MISMATCH maps_id={mid} ({len(mism)} tile(s)) in {args.zip.name}:")
    for m in mism[:200]:
        if m.csv_tid is None:
            print(f"  row={m.row} col={m.col} PHP terrain_id={m.php_tid} (OOB vs CSV)")
        else:
            print(
                f"  row={m.row} col={m.col} csv_terrain_id={m.csv_tid} "
                f"php_building_terrain_id={m.php_tid}"
            )
    if len(mism) > 200:
        print(f"  … {len(mism) - 200} more")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
