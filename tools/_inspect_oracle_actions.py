"""One-off: extract one sample of each action JSON from the oracle replay."""
import zipfile, gzip, json
from pathlib import Path

Z = Path(r"D:\Users\phili\AppData\Roaming\AWBWReplayPlayer\ReplayData\Replays\1630459.zip")

with zipfile.ZipFile(Z) as zf:
    blob = zf.read("a1630459")
txt = gzip.decompress(blob).decode("utf-8", errors="replace")

found: dict[str, str] = {}
i = 0
while i < len(txt) and len(found) < 10:
    j = txt.find("s:", i)
    if j < 0:
        break
    k = txt.find(":", j + 2)
    try:
        n = int(txt[j + 2 : k])
    except Exception:
        i = j + 2
        continue
    if txt[k + 1] != '"':
        i = j + 2
        continue
    start = k + 2
    body = txt[start : start + n]
    if body.startswith('{"action":'):
        try:
            obj = json.loads(body)
            atype = obj.get("action")
            if atype not in found:
                found[atype] = body
        except Exception:
            pass
    i = start + n + 2

for atype, body in found.items():
    print(f"=== {atype} ===")
    try:
        obj = json.loads(body)
        print(json.dumps(obj, indent=2)[:3500])
    except Exception as e:
        print("PARSE ERR:", e)
        print(body[:1000])
    print()

# Also print the first line's full structure (everything before first \n)
first = txt.split("\n", 1)[0]
print("=== first p: line (truncated) ===")
print(first[:600])
