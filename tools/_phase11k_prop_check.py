#!/usr/bin/env python3
"""Check engine props near (7,9) for gid 1635679."""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from engine.game import make_initial_state
from engine.map_loader import load_map
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game

CATS = [ROOT / "data" / "amarriner_gl_std_catalog.json",
        ROOT / "data" / "amarriner_gl_extras_catalog.json",
        ROOT / "data" / "amarriner_gl_colin_batch.json"]
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"

by_id = {}
for cp in CATS:
    if not cp.exists():
        continue
    cat = json.loads(cp.read_text(encoding="utf-8"))
    for g in (cat.get("games") or {}).values():
        if isinstance(g, dict) and "games_id" in g:
            by_id[int(g["games_id"])] = g
meta = by_id[1635679]
random.seed(_seed_for_game(CANONICAL_SEED, 1635679))
co0, co1 = pair_catalog_cos_ids(meta)
map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
state = make_initial_state(map_data, co0, co1, starting_funds=0,
                           tier_name=str(meta.get("tier") or "T2"))

# Show all properties near (7,9)
for p in state.properties:
    if abs(p.row - 7) <= 2 and abs(p.col - 9) <= 2:
        print(f"prop ({p.row},{p.col}) terrain={p.terrain_id} owner={p.owner} cap={p.capture_points}")

# Show map terrain near (7,9)
print()
for r in range(5, 10):
    line = []
    for c in range(7, 13):
        line.append(f"{map_data.terrain[r][c]:>4}")
    print(f"row{r}: " + "".join(line))
