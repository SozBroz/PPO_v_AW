"""Quick diagnostic for a map's income property layout.

Usage:
    python tools/diag_map_income.py 126428
    python tools/diag_map_income.py 126428 --co0 8 --co1 7

Prints:
  * Distinct terrain ids on the map.
  * Every property tile with classification (HQ/base/lab/comm/city/...).
  * Country -> player assignment (order of appearance).
  * After make_initial_state: funds[] + count_income_properties() per player.

Intended to answer "why does Sami open at 3k instead of 4k in replay 135015?"
Truth-table:
  engine 3k & AWBW 4k  -> map CSV missing HQ / mis-tagged -> fix map data.
  engine 4k & zip  3k  -> export/serializer loses 1000g    -> fix exporter.
  engine 4k & zip  4k but viewer shows 3k -> ReplaySetupContext bug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `engine` importable when this file is run directly.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine.terrain import get_terrain, get_country
from engine.map_loader import load_map
from engine.game import make_initial_state


POOL = REPO / "data" / "gl_map_pool.json"
MAPS_DIR = REPO / "data" / "maps"


def _classify(info) -> str:
    if info.is_hq:          return "HQ"
    if info.is_lab:         return "lab"
    if info.is_comm_tower:  return "comm"
    if info.is_base:        return "base"
    if info.is_airport:     return "airport"
    if info.is_port:        return "port"
    return "city"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map_id", type=int)
    ap.add_argument("--co0", type=int, default=1)
    ap.add_argument("--co1", type=int, default=7)
    args = ap.parse_args()

    md = load_map(args.map_id, POOL, MAPS_DIR)
    print(f"Map {args.map_id}: {md.name}  ({md.height}x{md.width})")
    print(f"Unit bans: {md.unit_bans}")

    # Distinct terrain ids
    tids = sorted({t for row in md.terrain for t in row})
    print(f"\nDistinct terrain_ids ({len(tids)}): {tids}")

    # Property tiles
    print("\nProperty tiles (by country):")
    by_country: dict = {}
    for r, row in enumerate(md.terrain):
        for c, tid in enumerate(row):
            info = get_terrain(tid)
            if not info.is_property:
                continue
            cid = get_country(tid)
            by_country.setdefault(cid, []).append((r, c, tid, _classify(info)))
    for cid, items in sorted(by_country.items(), key=lambda kv: (kv[0] is None, kv[0] or -1)):
        label = f"country {cid}" if cid is not None else "neutral"
        kinds = {}
        for _, _, _, k in items:
            kinds[k] = kinds.get(k, 0) + 1
        print(f"  {label:10s}  {len(items):3d} tiles   kinds={kinds}")

    print(f"\ncountry_to_player: {md.country_to_player}")
    print(f"objective_type:    {md.objective_type}")
    print(f"hq_positions:      {md.hq_positions}")
    print(f"lab_positions:     {md.lab_positions}")

    # Initial state
    state = make_initial_state(md, args.co0, args.co1, starting_funds=0, tier_name="T2")
    print("\nAfter make_initial_state(co0={}, co1={}):".format(args.co0, args.co1))
    print(f"  turn={state.turn}  active_player={state.active_player}")
    for p in (0, 1):
        props = [pr for pr in state.properties if pr.owner == p]
        income_props = state.count_income_properties(p)
        print(f"  P{p}: funds={state.funds[p]:6d}  owned={len(props):2d}  "
              f"income_props={income_props}  => grant={income_props * 1000}g")
    return 0


if __name__ == "__main__":
    sys.exit(main())
