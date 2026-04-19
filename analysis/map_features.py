"""
Extracts structural features from AWBW map CSV files.
Writes results to data/map_features.json.
"""
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
OUT_PATH = ROOT / "data" / "map_features.json"

# Terrain ID categories (approximate — matches engine/terrain.py)
SEA_IDS = set(range(28, 30))  # 28=sea, 29=shoal
PROPERTY_IDS = set(range(34, 170))  # all capturable properties
HQ_IDS = {42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117, 122, 127}
LAB_IDS = {133}  # neutral lab; 134+ are country labs
MOUNTAIN_IDS = {2}
WOOD_IDS = {3}
ROAD_IDS = set(range(14, 18))
PLAIN_IDS = {1}


def _load_csv(map_id: int) -> list[list[int]]:
    path = MAPS_DIR / f"{map_id}.csv"
    terrain = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                terrain.append([int(x) for x in line.split(",")])
    return terrain


def _is_property(tid: int) -> bool:
    return 34 <= tid <= 180  # loose range covering all country properties


def _is_hq(tid: int) -> bool:
    if tid < 42:
        return False
    return (tid - 42) % 5 == 0 and tid <= 162


def _is_lab(tid: int) -> bool:
    return tid == 133 or (134 <= tid <= 161 and (tid - 133) % 7 == 0)


def _is_sea(tid: int) -> bool:
    return tid == 28


def _bfs_distance(grid: list[list[int]], start: tuple, end: tuple) -> float:
    """Manhattan distance (fast approximation for map analysis)."""
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def extract_features(map_id: int, meta: dict) -> dict:
    """Extract structural features from a map's terrain grid."""
    terrain = _load_csv(map_id)
    H = len(terrain)
    W = len(terrain[0]) if terrain else 0
    total_tiles = H * W

    sea_count = 0
    property_count = 0
    hq_positions: list[tuple[int, int]] = []
    lab_positions: list[tuple[int, int]] = []
    mountain_count = 0
    wood_count = 0
    road_count = 0
    plain_count = 0

    for r, row in enumerate(terrain):
        for c, tid in enumerate(row):
            if _is_sea(tid):
                sea_count += 1
            if _is_property(tid):
                property_count += 1
            if _is_hq(tid):
                hq_positions.append((r, c))
            if _is_lab(tid):
                lab_positions.append((r, c))
            if tid == 2:
                mountain_count += 1
            if tid == 3:
                wood_count += 1
            if 14 <= tid <= 17:
                road_count += 1
            if tid == 1:
                plain_count += 1

    # Chokepoint density: count tiles where only 1-2 adjacent non-sea tiles exist
    choke_count = 0
    for r in range(H):
        for c in range(W):
            if _is_sea(terrain[r][c]):
                continue
            adjacent_land = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not _is_sea(terrain[nr][nc]):
                    adjacent_land += 1
            if adjacent_land <= 2:
                choke_count += 1

    objectives = hq_positions if hq_positions else lab_positions
    obj_distance = None
    if len(objectives) >= 2:
        obj_distance = _bfs_distance(terrain, objectives[0], objectives[-1])

    objective_type = "hq" if hq_positions else ("lab" if lab_positions else None)

    return {
        "map_id": map_id,
        "name": meta["name"],
        "type": meta["type"],
        "height": H,
        "width": W,
        "total_tiles": total_tiles,
        "sea_ratio": sea_count / total_tiles if total_tiles > 0 else 0.0,
        "property_count": property_count,
        "mountain_ratio": mountain_count / total_tiles if total_tiles > 0 else 0.0,
        "wood_ratio": wood_count / total_tiles if total_tiles > 0 else 0.0,
        "road_ratio": road_count / total_tiles if total_tiles > 0 else 0.0,
        "plain_ratio": plain_count / total_tiles if total_tiles > 0 else 0.0,
        "choke_density": choke_count / total_tiles if total_tiles > 0 else 0.0,
        "objective_type": objective_type,
        "objective_count": len(objectives),
        "hq_positions": hq_positions,
        "lab_positions": lab_positions,
        "obj_distance": obj_distance,
        "cap_limit": meta.get("cap_limit"),
        "unit_limit": meta.get("unit_limit"),
        "unit_bans": meta.get("unit_bans", []),
        "enabled_tiers": [t["tier_name"] for t in meta.get("tiers", []) if t.get("enabled")],
    }


def compute_all_features() -> dict:
    """Compute features for all maps in the pool."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(POOL_PATH) as f:
        pool = json.load(f)

    results = {}
    for meta in pool:
        map_id = meta["map_id"]
        csv_path = MAPS_DIR / f"{map_id}.csv"
        if not csv_path.exists():
            print(f"[map_features] Skipping {map_id} — CSV not found")
            continue
        try:
            features = extract_features(map_id, meta)
            results[str(map_id)] = features
            print(
                f"[map_features] {map_id} ({meta['name']}): "
                f"{features['height']}x{features['width']}, "
                f"sea={features['sea_ratio']:.2f}, "
                f"props={features['property_count']}, "
                f"obj={features['objective_type']}"
            )
        except Exception as e:
            print(f"[map_features] ERROR on {map_id}: {e}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[map_features] Written to {OUT_PATH}")
    return results


if __name__ == "__main__":
    compute_all_features()
