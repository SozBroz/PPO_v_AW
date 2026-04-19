"""Check co_max_power / co_max_spower across ALL turn snapshots."""
import gzip, io, re, zipfile
from pathlib import Path
import glob, sys

replays = sorted(glob.glob(r"C:\Users\phili\AWBW\replays\1?????.zip"))
path = Path(replays[-1])
print(f"Checking: {path}\n")

with zipfile.ZipFile(path) as z:
    raw = z.read(z.namelist()[0])
with gzip.open(io.BytesIO(raw)) as gz:
    text = gz.read().decode("utf-8")

lines = [l for l in text.split("\n") if l.strip()]
print(f"Total turn snapshots: {len(lines)}\n")

player_pat = re.compile(r'O:10:"awbwPlayer":\d+:\{')

problems = []

for turn_idx, line in enumerate(lines):
    for pi, m in enumerate(player_pat.finditer(line)):
        chunk = line[m.end():]
        for field in ["co_max_power", "co_max_spower"]:
            fp = f's:{len(field.encode())}:"{field}";'
            fi = chunk.find(fp)
            if fi >= 0:
                rest = chunk[fi+len(fp):]
                val = rest[:15].split(";")[0]
                if val == "N" or val == "i:0":
                    problems.append(f"  turn={turn_idx} player={pi} {field}={val}")

if problems:
    print("PROBLEMS found:")
    for p in problems:
        print(p)
else:
    print("All co_max_power / co_max_spower values are non-null and non-zero across all turns. OK")

# Also check if co_id changes between turns (would trigger onCOChange)
print("\n--- CO id per turn (all players) ---")
last_co = {}
for turn_idx, line in enumerate(lines):
    for pi, m in enumerate(player_pat.finditer(line)):
        chunk = line[m.end():]
        fp = 's:5:"co_id";'
        fi = chunk.find(fp)
        if fi >= 0:
            rest = chunk[fi+len(fp):]
            val = rest[:10].split(";")[0]
            key = f"p{pi}"
            if key not in last_co or last_co[key] != val:
                print(f"  turn={turn_idx} player={pi} co_id={val} {'<-- CHANGED' if key in last_co else ''}")
                last_co[key] = val
