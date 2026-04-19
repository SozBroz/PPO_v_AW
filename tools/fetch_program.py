import urllib.request, base64, json

def fetch_blob(sha):
    url = f"https://api.github.com/repos/DeamonHunter/AWBW-Replay-Player/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

src = fetch_blob("fa773b7ec6eda47bd0b996a9b6ee4e0a6671a8e1")
print("=== Program.cs ===")
safe = "".join(c if ord(c) < 128 else "?" for c in src[:3000])
print(safe)
