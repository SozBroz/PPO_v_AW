"""
Rewrite ``data/maps/<map_id>.csv`` so the two competitive factions use **Orange Star** and
**Blue Moon** terrain IDs, and set ``p0_country_id`` to ``1`` in ``gl_map_pool.json``.

Run after adding a map or when ingesting replays for debugging so seating matches a fixed
OS/BM convention::

  python tools/normalize_map_to_os_bm.py --map-id 140000 --dry-run
  python tools/normalize_map_to_os_bm.py --map-id 140000
  python tools/normalize_map_to_os_bm.py --from-catalog

After a successful normalize (non-dry-run), ``data/maps/<map_id>_units.json`` is
**re-written** from :func:`engine.map_loader.load_map` so predeploy ``player`` /
``force_engine_player`` match OS/BM terrain + ``p0_country_id: 1``. In-replay **built**
units live only in zip PHP snapshots — this tool does not rewrite those.

Import :func:`run_normalize_map_to_os_bm` from download tools to normalize immediately
after each successful replay zip write.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.map_country_normalize import (  # noqa: E402
    engine_seat_country_pair,
    normalize_terrain_grid,
)
from engine.map_loader import load_map  # noqa: E402
from engine.predeployed import PredeployedUnitSpec  # noqa: E402


@dataclass
class NormalizeMapResult:
    ok: bool
    map_id: int
    changed_cells: int = 0
    engine_p0_country: int | None = None
    engine_p1_country: int | None = None
    wrote_csv: bool = False
    updated_pool: bool = False
    updated_units_json: bool = False
    dry_run: bool = False
    error: str | None = None


def _load_csv(path: Path) -> list[list[int]]:
    terrain: list[list[int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                terrain.append([int(x) for x in line.split(",")])
    return terrain


def _write_csv(path: Path, terrain: list[list[int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in terrain:
            f.write(",".join(str(x) for x in row))
            f.write("\n")


def _build_units_json_payload(
    specs: list[PredeployedUnitSpec],
    preserved_top_level: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild ``*_units.json`` body; keep ``_source`` and other non-``units`` keys."""
    rows: list[dict[str, Any]] = []
    for s in specs:
        row: dict[str, Any] = {
            "row": s.row,
            "col": s.col,
            "player": s.player,
            "unit_type": s.unit_type.name,
            "hp": s.hp,
        }
        if s.force_engine_player is not None:
            row["force_engine_player"] = int(s.force_engine_player)
        rows.append(row)
    out = dict(preserved_top_level)
    sv = preserved_top_level.get("schema_version", 1)
    try:
        out["schema_version"] = int(sv)
    except (TypeError, ValueError):
        out["schema_version"] = 1
    out["units"] = rows
    return out


def _reconcile_predeployed_units_json(
    map_id: int,
    *,
    maps_dir: Path,
    pool_path: Path,
    backup: bool,
) -> bool:
    """Rewrite ``<map_id>_units.json`` from ``load_map`` after CSV/pool OS/BM sync."""
    units_path = maps_dir / f"{map_id}_units.json"
    if not units_path.is_file():
        return False
    try:
        map_data = load_map(map_id, pool_path, maps_dir)
    except Exception as exc:
        warnings.warn(
            f"normalize_map_to_os_bm(map_id={map_id}): left {units_path.name} unchanged "
            f"(post-normalize load_map / reconcile failed: {exc})",
            UserWarning,
            stacklevel=2,
        )
        return False
    raw_prev: dict[str, Any] = json.loads(units_path.read_text(encoding="utf-8"))
    preserved = {k: v for k, v in raw_prev.items() if k != "units"}
    payload = _build_units_json_payload(map_data.predeployed_specs, preserved)
    new_text = json.dumps(payload, indent=2) + "\n"
    old_text = units_path.read_text(encoding="utf-8")
    if new_text == old_text:
        return False
    if backup:
        bak = units_path.with_suffix(".json.bak")
        shutil.copy2(units_path, bak)
    units_path.write_text(new_text, encoding="utf-8")
    return True


def _pool_map_entry(pool: list, map_id: int) -> dict | None:
    for m in pool:
        if isinstance(m, dict) and int(m.get("map_id", 0)) == map_id:
            return m
    return None


def catalog_map_ids(catalog_path: Path) -> Iterator[int]:
    """Yield unique map_id values from ``amarriner_gl_std_catalog.json``."""
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    seen: set[int] = set()
    for g in (data.get("games") or {}).values():
        if not isinstance(g, dict):
            continue
        mid = g.get("map_id")
        if mid is None:
            continue
        i = int(mid)
        if i not in seen:
            seen.add(i)
            yield i


def run_normalize_map_to_os_bm(
    map_id: int,
    *,
    maps_dir: Path | None = None,
    pool_path: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> NormalizeMapResult:
    """
    Remap competitive map terrain to Orange Star / Blue Moon and set ``p0_country_id`` to ``1``.

    Safe to call repeatedly (idempotent when already normalized). Call after downloading
    a replay zip, **before** desync_audit or other tooling that loads ``load_map``.
    """
    maps_dir = maps_dir or (ROOT / "data" / "maps")
    pool_path = pool_path or (ROOT / "data" / "gl_map_pool.json")
    out = NormalizeMapResult(ok=False, map_id=map_id, dry_run=dry_run)

    csv_path = maps_dir / f"{map_id}.csv"
    if not csv_path.is_file():
        out.error = f"missing {csv_path}"
        return out

    try:
        pool_data: list[Any] = json.loads(pool_path.read_text(encoding="utf-8"))
    except OSError as e:
        out.error = str(e)
        return out
    meta = _pool_map_entry(pool_data, map_id)
    if meta is None:
        out.error = f"map_id {map_id} not in pool"
        return out

    raw_p0 = meta.get("p0_country_id")
    if raw_p0 is None:
        out.error = "pool entry has no p0_country_id"
        return out
    p0_cid = int(raw_p0)

    terrain = _load_csv(csv_path)
    pair = engine_seat_country_pair(terrain, p0_cid)
    if pair is None:
        out.error = "could not infer two country IDs (grid + p0_country_id)"
        return out
    eng_p0, eng_p1 = pair
    out.engine_p0_country = eng_p0
    out.engine_p1_country = eng_p1

    new_grid = normalize_terrain_grid(
        terrain,
        engine_p0_country_id=eng_p0,
        engine_p1_country_id=eng_p1,
    )
    changed = sum(
        1
        for r in range(len(terrain))
        for c in range(len(terrain[r]))
        if terrain[r][c] != new_grid[r][c]
    )
    out.changed_cells = changed

    if dry_run:
        out.ok = True
        return out

    if changed > 0:
        if backup:
            bak = csv_path.with_suffix(".csv.bak")
            shutil.copy2(csv_path, bak)
        _write_csv(csv_path, new_grid)
        out.wrote_csv = True

    if p0_cid != 1:
        meta["p0_country_id"] = 1
        pool_path.write_text(json.dumps(pool_data, indent=2) + "\n", encoding="utf-8")
        out.updated_pool = True

    if _reconcile_predeployed_units_json(
        map_id, maps_dir=maps_dir, pool_path=pool_path, backup=backup
    ):
        out.updated_units_json = True

    out.ok = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-id", type=int, default=None, help="Single map to normalize")
    ap.add_argument(
        "--from-catalog",
        action="store_true",
        help="Normalize every unique map_id in amarriner_gl_std_catalog.json (skips missing csv)",
    )
    ap.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "data" / "amarriner_gl_std_catalog.json",
        help="Catalog JSON for --from-catalog",
    )
    ap.add_argument(
        "--maps-dir",
        type=Path,
        default=ROOT / "data" / "maps",
        help="Directory containing <map_id>.csv",
    )
    ap.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "data" / "gl_map_pool.json",
        help="gl_map_pool.json path",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write files",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write <map_id>.csv.bak before overwrite",
    )
    args = ap.parse_args()

    if args.from_catalog and args.map_id is not None:
        print("Use either --map-id or --from-catalog, not both", file=sys.stderr)
        return 1
    if not args.from_catalog and args.map_id is None:
        ap.error("must pass --map-id or --from-catalog")

    ids: list[int]
    if args.from_catalog:
        if not args.catalog.is_file():
            print(f"missing {args.catalog}", file=sys.stderr)
            return 1
        ids = sorted(catalog_map_ids(args.catalog))
        print(f"[normalize] from-catalog: {len(ids)} unique map_ids")
    else:
        ids = [int(args.map_id)]

    failed = 0
    for mid in ids:
        res = run_normalize_map_to_os_bm(
            mid,
            maps_dir=args.maps_dir,
            pool_path=args.pool,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        if not res.ok:
            print(f"map_id={mid} SKIP: {res.error}", file=sys.stderr)
            failed += 1
            continue
        msg = (
            f"map_id={mid}: P0cid={res.engine_p0_country} P1cid={res.engine_p1_country} "
            f"cells_changed={res.changed_cells}"
        )
        if res.dry_run:
            msg += " (dry-run)"
        elif res.changed_cells == 0 and res.engine_p0_country is not None:
            msg += " (already OS/BM + p0_country_id=1)"
        if res.updated_units_json:
            msg += " predeploy_units_json=updated"
        print(msg)

    if failed:
        print(f"[normalize] done with {failed} failures (missing csv / not in pool / infer error)")
        return 1 if failed == len(ids) else 0
    print("[normalize] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
