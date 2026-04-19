"""Fetch multiple blobs from the AWBW viewer repo and save them."""
import urllib.request, base64, json, sys

def fetch_blob(sha, label):
    url = f"https://api.github.com/repos/DeamonHunter/AWBW-Replay-Player/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    src = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    print(f"\n{'='*80}")
    print(f"=== {label} ===")
    print(f"{'='*80}")
    # Print ascii-safe
    safe = "".join(c if ord(c) < 128 else "?" for c in src)
    print(safe)

blobs = [
    ("8a8fc0f6f2ef3f7257b9d48dde97fe04646b4f92", "Exceptions/ReplayParseExceptions.cs"),
    ("a1153707f65239d0ee7d48dff3a73b2683a5155b", "API/Replay/AWBWJsonReplayParser.cs"),
    ("def2cfd85bb2803a7a325285d6421639c37131b7", "API/Replay/ReplayMap.cs"),
    ("1680ea6522e42429f9ec812ff1698166b5fd0c32", "API/Replay/ReplayData.cs"),
]

for sha, label in blobs:
    fetch_blob(sha, label)
