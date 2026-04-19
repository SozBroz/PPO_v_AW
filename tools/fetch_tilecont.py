import urllib.request, base64, json

def fetch_blob(sha, label):
    url = f"https://api.github.com/repos/DeamonHunter/AWBW-Replay-Player/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    src = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return "".join(c if ord(c) < 128 else "?" for c in src)

# TileGridContainer likely has AddTile with the gridPosition check
blobs = [
    ("5d4813fd5912fb5da4d5c872a8479f93b897eb72", "UI/Components/TileGridContainer.cs"),
    ("40bc76ae2662e9a530ea435b6b2a870606cd9915", "API/Replay/ReplayPostProcessor.cs"),
]

for sha, label in blobs:
    src = fetch_blob(sha, label)
    print(f"\n{'='*80}\n=== {label} ===\n{'='*80}")
    print(src)
