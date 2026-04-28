import gzip, io, re, zipfile, json
from pathlib import Path

NEW_ZIP = Path(r"D:\AWBW\replays\128692.zip")
TERRAIN = Path(r"D:\Users\phili\AppData\Roaming\AWBWReplayPlayer\ReplayData\Terrain\180298.json")


def load_line0(path):
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


def extract_positions(line, cls):
    pat = re.compile(rf'O:{len(cls)}:"{re.escape(cls)}":\d+:\{{')
    results = []
    for m in pat.finditer(line):
        chunk = line[m.end():]
        d = {}
        for field in ["x", "y"]:
            fp = f's:{len(field)}:"{field}";'
            fi = chunk.find(fp)
            if fi >= 0:
                rest = chunk[fi + len(fp):]
                v = rest[:15].split(";")[0]
                if v.startswith("i:"):
                    d[field] = int(v[2:])
        results.append(d)
    return results


line = load_line0(NEW_ZIP)

if TERRAIN.exists():
    t = json.loads(TERRAIN.read_text())
    w, h = t["Size"]["X"], t["Size"]["Y"]
    print(f"Map 180298: {w} cols x {h} rows  (0-based [0..{w-1}] x [0..{h-1}])")
else:
    # extract from replay awbwMap header
    m = re.search(r's:4:"size";O:\d+:"awbwSize":\d+:\{[^}]*s:1:"x";i:(\d+);s:1:"y";i:(\d+)', line)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        print(f"Map 180298 (from replay header): {w} cols x {h} rows")
    else:
        print("Could not determine map dimensions. Terrain JSON not found locally.")
        w, h = 999, 999

buildings = extract_positions(line, "awbwBuilding")
units = extract_positions(line, "awbwUnit")

bx = [b["x"] for b in buildings if "x" in b]
by = [b["y"] for b in buildings if "y" in b]
oob = [(x, y) for x, y in zip(bx, by) if x < 0 or x >= w or y < 0 or y >= h]
print(f"Buildings: {len(buildings)}  x:[{min(bx)}..{max(bx)}]  y:[{min(by)}..{max(by)}]  OOB: {oob}")

if units:
    ux = [u["x"] for u in units if "x" in u]
    uy = [u["y"] for u in units if "y" in u]
    oob_u = [(x, y) for x, y in zip(ux, uy) if x < 0 or x >= w or y < 0 or y >= h]
    print(f"Units: {len(units)}  x:[{min(ux)}..{max(ux)}]  y:[{min(uy)}..{max(uy)}]  OOB: {oob_u}")
else:
    print("Units: 0")
