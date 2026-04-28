"""Check actual co_max_power values in the generated replay for each player."""
import gzip, io, re, zipfile
from pathlib import Path
import sys, os
sys.path.insert(0, r"D:\AWBW")

# Find latest generated replay
import glob
replays = sorted(glob.glob(r"D:\AWBW\replays\1?????.zip"))
if not replays:
    print("No generated replays found"); sys.exit(1)
path = Path(replays[-1])
print(f"Checking: {path}")

with zipfile.ZipFile(path) as z:
    raw = z.read(z.namelist()[0])
with gzip.open(io.BytesIO(raw)) as gz:
    line = gz.read().decode("utf-8").split("\n")[0]

# Extract all awbwPlayer objects
player_pat = re.compile(r'O:10:"awbwPlayer":\d+:\{')
for pi, m in enumerate(player_pat.finditer(line)):
    chunk = line[m.end():]
    print(f"\nPlayer {pi}:")
    for field in ["id", "co_id", "co_power", "co_max_power", "co_max_spower",
                  "tags_co_max_power", "tags_co_max_spower"]:
        fp = f's:{len(field.encode())}:"{field}";'
        fi = chunk.find(fp)
        if fi >= 0:
            rest = chunk[fi+len(fp):]
            val = rest[:30].split(";")[0]
            print(f"  {field:22s}: {val};")
        else:
            print(f"  {field:22s}: <MISSING>")
