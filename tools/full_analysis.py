"""
Full analysis of coordinate system and building mismatch.
Phases B, C, D in one script.
"""
import gzip, io, re, zipfile, json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, r"D:\AWBW")

TERRAIN_133665 = Path(r"D:\Users\phili\AppData\Roaming\AWBWReplayPlayer\ReplayData\Terrain\133665.json")
TERRAIN_173170 = Path(r"D:\Users\phili\AppData\Roaming\AWBWReplayPlayer\ReplayData\Terrain\173170.json")
REF_ZIP  = Path(r"D:\AWBW\replays\replay_1630459_GL_STD_[T1]__Gronktastic_vs_justbored_2026-04-16.zip")
import glob
gen = sorted(glob.glob(r"D:\AWBW\replays\1?????.zip"))
CAND_ZIP = Path(gen[-1]) if gen else None


def load_line0(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            raw = z.read(name)
            try:
                with gzip.open(io.BytesIO(raw)) as gz:
                    text = gz.read().decode("utf-8")
            except Exception:
                text = raw.decode("utf-8", errors="replace")
            if "awbwGame" in text:
                return text.split("\n")[0]
    raise RuntimeError(f"No awbwGame in {path}")


def extract_cls(line, cls):
    pat = re.compile(rf'O:{len(cls)}:"{re.escape(cls)}":\d+:\{{')
    results = []
    for m in pat.finditer(line):
        chunk = line[m.end():]
        d = {}
        for field in ["id", "x", "y", "terrain_id", "name", "players_id", "capture"]:
            fp = f's:{len(field.encode())}:"{field}";'
            fi = chunk.find(fp)
            if fi >= 0:
                rest = chunk[fi+len(fp):]
                raw_val = rest[:25].split(";")[0]
                if raw_val.startswith("i:"):
                    d[field] = int(raw_val[2:])
                elif raw_val.startswith("s:"):
                    m2 = re.search(r'"([^"]+)"', raw_val)
                    d[field] = m2.group(1) if m2 else raw_val
                elif raw_val == "N":
                    d[field] = None
                else:
                    d[field] = raw_val
        results.append(d)
    return results


# Phase B: Map dimensions
print("=" * 70)
print("PHASE B: Map dimensions")
print("=" * 70)

t133 = json.loads(TERRAIN_133665.read_text())
w, h = t133["Size"]["X"], t133["Size"]["Y"]
print(f"  Terrain 133665: X={w} (cols), Y={h} (rows)")
print(f"  Viewer 0-based grid: x in [0..{w-1}], y in [0..{h-1}]")
print(f"  Engine ai_vs_ai printed: 21x25 (rows x cols)")
print(f"  Match: {'YES' if h == 21 and w == 25 else 'NO - MISMATCH!'}")

print()

# Phase C: Position comparison
print("=" * 70)
print("PHASE C: Building and unit positions")
print("=" * 70)

ref_line  = load_line0(REF_ZIP)
ref_buildings = extract_cls(ref_line, "awbwBuilding")
ref_units     = extract_cls(ref_line, "awbwUnit")

print(f"\n--- Reference replay (maps_id=173170) ---")
ref_t = json.loads(TERRAIN_173170.read_text()) if TERRAIN_173170.exists() else None
if ref_t:
    rw, rh = ref_t["Size"]["X"], ref_t["Size"]["Y"]
else:
    rw, rh = 999, 999

bx = [b["x"] for b in ref_buildings if b.get("x") is not None]
by = [b["y"] for b in ref_buildings if b.get("y") is not None]
print(f"  Buildings: {len(ref_buildings)}  x:[{min(bx)}..{max(bx)}]  y:[{min(by)}..{max(by)}]  0-based OOB: {sum(1 for x,y in zip(bx,by) if x<0 or x>=rw or y<0 or y>=rh)}")
print(f"  First 10 building positions: {[(b['x'],b['y']) for b in ref_buildings[:10]]}")
ux = [u["x"] for u in ref_units if u.get("x") is not None]
uy = [u["y"] for u in ref_units if u.get("y") is not None]
if ux:
    print(f"  Units: {len(ref_units)}  x:[{min(ux)}..{max(ux)}]  y:[{min(uy)}..{max(uy)}]")

if CAND_ZIP:
    cand_line = load_line0(CAND_ZIP)
    cand_buildings = extract_cls(cand_line, "awbwBuilding")
    cand_units     = extract_cls(cand_line, "awbwUnit")

    print(f"\n--- Homebrew replay: {CAND_ZIP.name} (maps_id=133665) ---")
    bx = [b["x"] for b in cand_buildings if b.get("x") is not None]
    by = [b["y"] for b in cand_buildings if b.get("y") is not None]
    oob_0 = [(b["x"],b["y"]) for b in cand_buildings if b.get("x") is not None and
              (b["x"]<0 or b["x"]>=w or b["y"]<0 or b["y"]>=h)]
    print(f"  Buildings: {len(cand_buildings)}  x:[{min(bx)}..{max(bx)}]  y:[{min(by)}..{max(by)}]")
    print(f"  0-based OOB positions: {oob_0}")
    print(f"  First 10 building positions: {[(b['x'],b['y']) for b in cand_buildings[:10]]}")
    ux = [u["x"] for u in cand_units if u.get("x") is not None]
    uy = [u["y"] for u in cand_units if u.get("y") is not None]
    if ux:
        oob_u = [(u["x"],u["y"]) for u in cand_units if u.get("x") is not None and
                  (u["x"]<0 or u["x"]>=w or u["y"]<0 or u["y"]>=h)]
        print(f"  Units: {len(cand_units)}  x:[{min(ux)}..{max(ux)}]  y:[{min(uy)}..{max(uy)}]  OOB: {oob_u}")

# Phase C: Building tile count from terrain JSON
print(f"\n{'='*70}")
print("PHASE C: Building tile count in terrain JSON vs replay")
print("=" * 70)

# Building terrain IDs — from AWBW conventions, buildings are properties
# Key IDs: 34=city, 112=airport, 120=port, 133=base, 145=HQ, 197=lab, 200=comtower
# Plus country-specific variants (34+country offsets etc.)
# Let's check which IDs in the terrain appear as terrainId in BOTH viewer's building storage check
# We can't call C# code, so use the IDs we saw in our exporter's terrain mapping

# From engine terrain.py - any terrain that is_property
# Let's load TERRAIN_TABLE and check
from engine.terrain import TERRAIN_TABLE
property_engine_ids = set()
for tid, info in TERRAIN_TABLE.items():
    if info.is_property:
        property_engine_ids.add(tid)

print(f"  Engine property terrain IDs: {sorted(property_engine_ids)}")

# Count tiles in map 133665 that match property IDs
t_ids = t133["Ids"]
property_tiles_in_map = [(i % w, i // w, tid) for i, tid in enumerate(t_ids) if tid in property_engine_ids]
print(f"  Property tiles in terrain JSON 133665: {len(property_tiles_in_map)}")
from collections import Counter
c = Counter(t for _, _, t in property_tiles_in_map)
print(f"  Breakdown by terrain_id: {dict(sorted(c.items()))}")

# From engine: load the actual GameState for map 133665 to see how many properties we generate
print(f"\n--- GameState properties for map 133665 ---")
try:
    from engine.game import make_initial_state
    from data.maps import load_map
    map_data = load_map(133665)
    state = make_initial_state(map_data)
    print(f"  engine map dimensions: rows={map_data.map_height}, cols={map_data.map_width}")
    print(f"  properties count: {len(state.properties)}")
    # Show first few
    for prop in state.properties[:5]:
        print(f"    row={prop.row}, col={prop.col}, terrain_id={prop.terrain_id}, owner={prop.owner}")
except Exception as e:
    print(f"  Error loading GameState: {e}")

# Phase D: action stream check
print(f"\n{'='*70}")
print("PHASE D: Action stream check")
print("=" * 70)
with zipfile.ZipFile(CAND_ZIP) as z:
    entries = z.namelist()
    print(f"  Zip entries: {entries}")
    has_actions = any(n.startswith("a") for n in entries)
    print(f"  Has action stream: {has_actions}")
