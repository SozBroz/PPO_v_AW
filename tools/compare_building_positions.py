"""
Extract and compare building (x, y) positions from reference vs homebrew replay.
Also compare unit positions.
"""
import gzip, io, re, zipfile, json
from pathlib import Path

# Reference replay
REF_PATH  = Path(r"D:\AWBW\replays\replay_1630459_GL_STD_[T1]__Gronktastic_vs_justbored_2026-04-16.zip")
# Latest homebrew replay
import glob
gen = sorted(glob.glob(r"D:\AWBW\replays\1?????.zip"))
CAND_PATH = Path(gen[-1]) if gen else None

# Terrain JSON for the homebrew map (133665)
TERRAIN_PATH = Path(r"D:\Users\phili\AppData\Roaming\AWBWReplayPlayer\ReplayData\Terrain\133665.json")


def load_turn0(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            raw = z.read(name)
            try:
                with gzip.open(io.BytesIO(raw)) as gz:
                    text = gz.read().decode("utf-8")
            except OSError:
                text = raw.decode("utf-8", errors="replace")
            if "awbwGame" in text:
                return text.split("\n")[0]
    raise RuntimeError(f"No awbwGame in {path}")


def extract_object_fields(line: str, cls: str):
    """Yield dict of fields for each object of class cls."""
    pat = re.compile(rf'O:{len(cls)}:"{re.escape(cls)}":\d+:\{{')
    for m in pat.finditer(line):
        chunk = line[m.end():]
        d = {}
        for field in ["id", "x", "y", "terrain_id", "players_id", "name",
                       "movement_points", "units_id"]:
            fp = f's:{len(field.encode())}:"{field}";'
            fi = chunk.find(fp)
            if fi >= 0:
                rest = chunk[fi+len(fp):]
                val = rest[:20].split(";")[0]
                # strip PHP type prefix
                if val.startswith("i:"):
                    d[field] = int(val[2:])
                elif val.startswith("s:"):
                    inner = val.split('"')
                    d[field] = inner[1] if len(inner) > 2 else val
                elif val == "N":
                    d[field] = None
                else:
                    d[field] = val
        yield d


print("Loading terrain JSON...")
terrain = json.loads(TERRAIN_PATH.read_text())
map_w = terrain["Size"]["X"]  # columns
map_h = terrain["Size"]["Y"]  # rows
print(f"  Map size: width={map_w} cols (X), height={map_h} rows (Y)")
print(f"  Viewer expects 0-based grid: x in [0..{map_w-1}], y in [0..{map_h-1}]")

print(f"\n=== Reference replay: {REF_PATH.name} ===")
ref_line = load_turn0(REF_PATH)
ref_buildings = list(extract_object_fields(ref_line, "awbwBuilding"))
print(f"  Buildings: {len(ref_buildings)}")
ref_xs = [b.get("x") for b in ref_buildings if b.get("x") is not None]
ref_ys = [b.get("y") for b in ref_buildings if b.get("y") is not None]
if ref_xs:
    print(f"  x range: {min(ref_xs)}..{max(ref_xs)}")
    print(f"  y range: {min(ref_ys)}..{max(ref_ys)}")
    # Check if any OOB for 0-based [0..width-1]
    oob_0 = [(b["x"], b["y"]) for b in ref_buildings if b.get("x") is not None
              and (b["x"] < 0 or b["y"] < 0 or b["x"] >= map_w or b["y"] >= map_h)]
    oob_1 = [(b["x"], b["y"]) for b in ref_buildings if b.get("x") is not None
              and (b["x"] < 1 or b["y"] < 1 or b["x"] > map_w or b["y"] > map_h)]
    print(f"  OOB if 0-based [0..{map_w-1}]: {len(oob_0)} buildings")
    print(f"  OOB if 1-based [1..{map_w}]: {len(oob_1)} buildings")
    print(f"  First 5 buildings: {[(b.get('x'), b.get('y'), b.get('terrain_id')) for b in ref_buildings[:5]]}")

ref_units = list(extract_object_fields(ref_line, "awbwUnit"))
if ref_units:
    unit_xs = [u.get("x") for u in ref_units if u.get("x") is not None]
    unit_ys = [u.get("y") for u in ref_units if u.get("y") is not None]
    print(f"  Units: {len(ref_units)}, x: {min(unit_xs)}..{max(unit_xs)}, y: {min(unit_ys)}..{max(unit_ys)}")

if CAND_PATH:
    print(f"\n=== Homebrew replay: {CAND_PATH.name} ===")
    cand_line = load_turn0(CAND_PATH)
    cand_buildings = list(extract_object_fields(cand_line, "awbwBuilding"))
    print(f"  Buildings: {len(cand_buildings)}")
    cand_xs = [b.get("x") for b in cand_buildings if b.get("x") is not None]
    cand_ys = [b.get("y") for b in cand_buildings if b.get("y") is not None]
    if cand_xs:
        print(f"  x range: {min(cand_xs)}..{max(cand_xs)}")
        print(f"  y range: {min(cand_ys)}..{max(cand_ys)}")
        oob_0 = [(b["x"], b["y"]) for b in cand_buildings if b.get("x") is not None
                  and (b["x"] < 0 or b["y"] < 0 or b["x"] >= map_w or b["y"] >= map_h)]
        oob_1 = [(b["x"], b["y"]) for b in cand_buildings if b.get("x") is not None
                  and (b["x"] < 1 or b["y"] < 1 or b["x"] > map_w or b["y"] > map_h)]
        print(f"  OOB if 0-based [0..{map_w-1}]: {len(oob_0)} buildings")
        print(f"  OOB if 1-based [1..{map_w}]: {len(oob_1)} buildings")
        print(f"  First 5 buildings: {[(b.get('x'), b.get('y'), b.get('terrain_id')) for b in cand_buildings[:5]]}")

    cand_units = list(extract_object_fields(cand_line, "awbwUnit"))
    if cand_units:
        unit_xs = [u.get("x") for u in cand_units if u.get("x") is not None]
        unit_ys = [u.get("y") for u in cand_units if u.get("y") is not None]
        print(f"  Units: {len(cand_units)}, x: {min(unit_xs)}..{max(unit_xs)}, y: {min(unit_ys)}..{max(unit_ys)}")

# Also cross-check: which terrain tiles in map 133665 are buildings?
print(f"\n=== Terrain JSON property tiles for 133665 ===")
ids = terrain["Ids"]
# Property IDs for unowned: city=34, base=1/2..., HQ..., airport..., port..., lab...
# Let's just count tiles that are not plains(28)/sea/forest etc
# From viewer BuildingStorage - anything with CountryID != -1 or is a building
# For simplicity: show all non-plain terrain IDs
from collections import Counter
c = Counter(ids)
print(f"  Terrain ID frequencies: {dict(sorted(c.items()))}")
print(f"  Total tiles: {len(ids)} (expected {map_w*map_h}={map_w*map_h})")
