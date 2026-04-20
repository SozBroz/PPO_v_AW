"""
Validate OS/BM convention across pool, CSV, properties, predeploy, and catalog replay zips.

- **Pool:** every map with a CSV has ``p0_country_id == 1``.
- **Terrain:** competitive properties use only AWBW ``country_id`` 1 (Orange Star) and 2 (Blue Moon);
  ``MapData.country_to_player`` is ``{1: 0, 2: 1}`` (engine P0 = OS, P1 = BM).
- **Predeploy:** units on OS/BM property tiles belong to the matching engine seat (unless
  ``force_engine_player`` applies — covered by count + placement checks).
- **Catalog zips:** each ``replays/amarriner_gl/<games_id>.zip`` listed in the catalog parses as a
  PHP replay (gzip + serialize). Maps are validated above; **do not** expect PHP snapshot
  ``terrain_id`` values to be only OS/BM tile IDs — AWBW stores the live palette (many countries).
  Engine simulation uses normalized ``data/maps/<map_id>.csv`` instead.

Usage::

  python tools/validate_os_bm_consistency.py
  python tools/validate_os_bm_consistency.py --no-catalog-zips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from engine.predeployed import load_predeployed_units_file  # noqa: E402
from engine.terrain import get_country, get_terrain  # noqa: E402
from tools.diff_replay_zips import load_replay  # noqa: E402


def _validate_map(
    map_id: int,
    *,
    maps_dir: Path,
    pool_path: Path,
    co0: int = 1,
    co1: int = 1,
) -> list[str]:
    errs: list[str] = []
    meta_list = json.loads(pool_path.read_text(encoding="utf-8"))
    meta = next((m for m in meta_list if isinstance(m, dict) and int(m.get("map_id", 0)) == map_id), None)
    if meta is None:
        return [f"map_id={map_id}: not in pool"]
    p0 = meta.get("p0_country_id")
    if p0 is None:
        errs.append(f"map_id={map_id}: missing p0_country_id")
    elif int(p0) != 1:
        errs.append(f"map_id={map_id}: p0_country_id={p0} expected 1")

    csv_path = maps_dir / f"{map_id}.csv"
    if not csv_path.is_file():
        return errs

    md = load_map(map_id, pool_path, maps_dir)
    ctp = md.country_to_player
    if ctp != {1: 0, 2: 1}:
        errs.append(f"map_id={map_id}: country_to_player={ctp} expected {{1: 0, 2: 1}}")

    for r, row in enumerate(md.terrain):
        for c, tid in enumerate(row):
            t = get_terrain(tid)
            if not t.is_property:
                continue
            cid = get_country(tid)
            if cid is None:
                continue
            if cid not in (1, 2):
                errs.append(f"map_id={map_id}: tile ({r},{c}) tid={tid} country_id={cid} not OS/BM")

    # Predeploy file: specs resolve to same units as make_initial_state
    spec_path = maps_dir / f"{map_id}_units.json"
    if spec_path.is_file():
        try:
            st = make_initial_state(md, co0, co1, starting_funds=0, tier_name="T2")
        except Exception as e:
            errs.append(f"map_id={map_id}: make_initial_state {e}")
            return errs
        raw_specs = load_predeployed_units_file(spec_path)
        n_pre = sum(len(st.units[p]) for p in (0, 1))
        if n_pre != len(raw_specs):
            errs.append(
                f"map_id={map_id}: predeploy count {len(raw_specs)} vs state units {n_pre}"
            )
        for u in st.units[0] + st.units[1]:
            tid = md.terrain[u.pos[0]][u.pos[1]]
            t = get_terrain(tid)
            if t.is_property:
                gc = get_country(tid)
                if gc == 1 and int(u.player) != 0:
                    errs.append(
                        f"map_id={map_id}: unit on OS tile at {u.pos} but player {u.player}"
                    )
                if gc == 2 and int(u.player) != 1:
                    errs.append(
                        f"map_id={map_id}: unit on BM tile at {u.pos} but player {u.player}"
                    )

    return errs


def _validate_catalog_zips_parse(
    *,
    catalog_path: Path,
    zips_dir: Path,
) -> tuple[list[str], int]:
    """Return (errors, n_zips_checked)."""
    errs: list[str] = []
    if not catalog_path.is_file():
        return ([f"missing catalog {catalog_path}"], 0)
    cat = json.loads(catalog_path.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    n = 0
    for g in games.values():
        if not isinstance(g, dict):
            continue
        gid = int(g.get("games_id", 0))
        zpath = zips_dir / f"{gid}.zip"
        if not zpath.is_file():
            continue
        n += 1
        try:
            frames = load_replay(zpath)
        except Exception as e:
            errs.append(f"{zpath.name}: load_replay failed: {e}")
            continue
        if not frames:
            errs.append(f"{zpath.name}: empty replay")
    return errs, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "data" / "maps")
    ap.add_argument("--pool", type=Path, default=ROOT / "data" / "gl_map_pool.json")
    ap.add_argument("--catalog", type=Path, default=ROOT / "data" / "amarriner_gl_std_catalog.json")
    ap.add_argument("--zips-dir", type=Path, default=ROOT / "replays" / "amarriner_gl")
    ap.add_argument(
        "--no-catalog-zips",
        action="store_true",
        help="Skip parsing each catalog replay zip (maps-only validation)",
    )
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    map_ids = sorted(
        int(m["map_id"])
        for m in pool
        if isinstance(m, dict) and m.get("map_id") is not None and (args.maps_dir / f'{int(m["map_id"])}.csv').is_file()
    )

    all_errs: list[str] = []
    for mid in map_ids:
        all_errs.extend(_validate_map(mid, maps_dir=args.maps_dir, pool_path=args.pool))

    n_zips = 0
    if not args.no_catalog_zips:
        zip_errs, n_zips = _validate_catalog_zips_parse(
            catalog_path=args.catalog,
            zips_dir=args.zips_dir,
        )
        all_errs.extend(zip_errs)

    if all_errs:
        print("FAILURES:", len(all_errs), file=sys.stderr)
        for e in all_errs:
            print(e, file=sys.stderr)
        return 1

    extra = f", {n_zips} catalog zips parse OK" if n_zips else ""
    print(
        f"[validate] OK: {len(map_ids)} maps with CSV, pool p0_country_id=1, OS/BM terrain, predeploy consistent{extra}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
