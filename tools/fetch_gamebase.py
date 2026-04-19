import urllib.request, base64, json

def fetch_blob(sha):
    url = f"https://api.github.com/repos/DeamonHunter/AWBW-Replay-Player/git/blobs/{sha}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

src = fetch_blob("15675a79f403a836b632cb9f9f851fb98ca06a44")
safe = "".join(c if ord(c) < 128 else "?" for c in src)
# Find import / args handling
for kw in ["ImportFiles", "args", "CommandLine", "HandleArgs", "PresentFile", "fileImport"]:
    idx = safe.find(kw)
    if idx >= 0:
        print(f"=== {kw} @{idx} ===")
        print(safe[max(0,idx-50):idx+300])
        print()
