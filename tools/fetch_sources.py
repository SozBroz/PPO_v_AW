import urllib.request, base64, json

def fetch_blob(sha):
    url = f"https://api.github.com/repos/DeamonHunter/AWBW-Replay-Player/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    return base64.b64decode(data["content"]).decode("utf-8")

# PlayerInfo.cs
src = fetch_blob("3e02e2b1d9b9977ff40c5e8ab6360541d9cffc8e")
print("=== PlayerInfo: COInfo/ActiveCO related fields ===")
for i, line in enumerate(src.split("\n")):
    for kw in ["COInfo", "ActiveCO", "PowerRequired", "co_max", "MaxPower", "coMaxPower"]:
        if kw in line:
            print(f"  {i+1:4d}: {line.rstrip()}")
            break

print()
# COData.cs
src2 = fetch_blob("54e2984409b4ac51dffb3d81ad0bb06b2c4e1459")
print("=== COData.cs ===")
# First 100 lines only
for i, line in enumerate(src2.split("\n")[:120]):
    print(f"  {i+1:4d}: {line.rstrip()}")
