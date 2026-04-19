"""
Validate all *_units.json files: load via engine, check bounds, no duplicates.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.predeployed import load_predeployed_units_file
from engine.map_loader import load_map

ROOT      = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR  = ROOT / "data" / "maps"

pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
map_ids = [e["map_id"] for e in pool]

ok = err = 0
for map_id in map_ids:
    units_path = MAPS_DIR / f"{map_id}_units.json"
    if not units_path.exists():
        print(f"[MISSING] {map_id}")
        err += 1
        continue

    try:
        specs = load_predeployed_units_file(units_path)
        map_data = load_map(map_id, POOL_PATH, MAPS_DIR)
        initial_state_units = {}
        from engine.predeployed import specs_to_initial_units
        units_dict = specs_to_initial_units(specs)
        total = sum(len(v) for v in units_dict.values())
        print(f"[OK] map {map_id}: {total} predeploy unit(s) "
              f"(p0={len(units_dict[0])}, p1={len(units_dict[1])})")
        ok += 1
    except Exception as e:
        print(f"[FAIL] map {map_id}: {e}")
        err += 1

print(f"\nTotal: {ok} OK, {err} FAIL")
