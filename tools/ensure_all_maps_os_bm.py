"""
Ensure every ``gl_map_pool.json`` map with a local CSV has ``p0_country_id`` set and
terrain remapped to Orange Star / Blue Moon (engine P0 = OS, P1 = BM).

- If ``p0_country_id`` is missing: set to ``1`` when the grid only uses countries ``1`` and ``2``;
  otherwise set to the **first property country** in row-major scan (matches ``load_map`` default
  before pool override), then run :func:`run_normalize_map_to_os_bm`.

Usage::

  python tools/ensure_all_maps_os_bm.py --dry-run
  python tools/ensure_all_maps_os_bm.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.map_country_normalize import infer_two_country_ids_from_grid  # noqa: E402
from engine.terrain import get_country, get_terrain  # noqa: E402
from tools.normalize_map_to_os_bm import _load_csv, run_normalize_map_to_os_bm  # noqa: E402


def _property_country_set(terrain: list[list[int]]) -> set[int]:
    out: set[int] = set()
    for row in terrain:
        for tid in row:
            t = get_terrain(tid)
            if not t.is_property:
                continue
            c = get_country(tid)
            if c is not None:
                out.add(c)
    return out


def _infer_p0_country_id(terrain: list[list[int]]) -> int | None:
    """Default pool ``p0_country_id`` before OS/BM remap."""
    props = _property_country_set(terrain)
    if len(props) < 2:
        return None
    if props == {1, 2}:
        return 1
    pair = infer_two_country_ids_from_grid(terrain)
    if pair is None:
        return None
    return int(pair[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pool", type=Path, default=ROOT / "data" / "gl_map_pool.json")
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "data" / "maps")
    args = ap.parse_args()

    pool: list = json.loads(args.pool.read_text(encoding="utf-8"))
    updated_meta = 0
    for m in pool:
        if not isinstance(m, dict) or m.get("map_id") is None:
            continue
        mid = int(m["map_id"])
        csv_path = args.maps_dir / f"{mid}.csv"
        if not csv_path.is_file():
            continue
        terrain = _load_csv(csv_path)
        if m.get("p0_country_id") is not None:
            continue
        p0 = _infer_p0_country_id(terrain)
        if p0 is None:
            print(f"[ensure] map_id={mid} SKIP: fewer than 2 property countries", file=sys.stderr)
            continue
        print(f"[ensure] map_id={mid} set p0_country_id {p0} (was missing)")
        m["p0_country_id"] = p0
        updated_meta += 1

    if updated_meta and not args.dry_run:
        args.pool.write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")
        print(f"[ensure] wrote {args.pool} ({updated_meta} entries gained p0_country_id)")
    elif updated_meta and args.dry_run:
        print(f"[ensure] dry-run: would set p0 on {updated_meta} maps")

    # Normalize every map that has csv + pool entry
    norm_ok = norm_fail = 0
    for m in pool:
        if not isinstance(m, dict) or m.get("map_id") is None:
            continue
        mid = int(m["map_id"])
        if not (args.maps_dir / f"{mid}.csv").is_file():
            continue
        if m.get("p0_country_id") is None:
            print(f"[ensure] map_id={mid} SKIP normalize: still no p0_country_id", file=sys.stderr)
            norm_fail += 1
            continue
        if args.dry_run:
            print(f"[ensure] would normalize map_id={mid}")
            norm_ok += 1
            continue
        res = run_normalize_map_to_os_bm(
            mid,
            maps_dir=args.maps_dir,
            pool_path=args.pool,
            dry_run=False,
            backup=True,
        )
        if res.ok:
            norm_ok += 1
            if res.changed_cells > 0 or res.updated_pool:
                print(
                    f"[normalize] map_id={mid} cells={res.changed_cells} "
                    f"wrote_csv={res.wrote_csv} pool={res.updated_pool}"
                )
        else:
            norm_fail += 1
            print(f"[normalize] map_id={mid} FAIL: {res.error}", file=sys.stderr)

    print(f"[ensure] normalize ok={norm_ok} fail={norm_fail}")
    return 0 if norm_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
