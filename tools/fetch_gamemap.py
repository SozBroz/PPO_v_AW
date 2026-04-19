import urllib.request, base64, json

def fetch_blob(sha, label):
    url = f"https://api.github.com/repos/DeamonHunter/AWBW-Replay-Player/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    src = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    safe = "".join(c if ord(c) < 128 else "?" for c in src)
    return safe

blobs = [
    ("ade44d08af24517ed83161152686ec3f64085287", "Game/Logic/GameMap.cs"),
    ("07d6da56e2588dffc4f26b91cad0cbb68256fdfa", "Game/Logic/ReplayController.cs"),
    ("87bb6c89a59ed3897138d4ea242827626c139912", "Game/Logic/ReplaySetupContext.cs"),
]

for sha, label in blobs:
    src = fetch_blob(sha, label)
    print(f"\n{'='*80}\n=== {label} ===\n{'='*80}")
    print(src)
