import base64, json, re

# PowerProgress.cs
raw = open(r"C:\Users\phili\.cursor\projects\c-Users-phili-AWBW\agent-tools\5a4fb8f6-0295-40e0-acc0-2718d5d2cb82.txt").read()
data = json.loads(raw)

# PowerProgress blob
pp_blob = base64.b64decode("77u/dXNpbmcgU3lzdGVtOwp1c2luZyBTeXN0ZW0uQ29sbGVjdGlvbnMuR2VuZXJpYzsKdXNpbmcgQVdCV0FwcC5HYW1lLkhlbHBlcnM7CnVzaW5nIG9zdS5GcmFtZXdvcmsuQmluZGFibGVzOwp1c2luZyBvc3UuRnJhbWV3b3JrLkV4dGVuc2lvbnMuQ29sb3I0RXh0ZW5zaW9uczsKdXNpbmcgb3N1LkZyYW1ld29yay5HcmFwaGljczsKdXNpbmcgb3N1LkZyYW1ld29yay5HcmFwaGljcy5Db250YWluZXJzOwp1c2luZyBvc3UuRnJhbWV3b3JrLkdyYXBoaWNzLkN1cnNvcjsKdXNpbmcgb3N1LkZyYW1ld29yay5HcmFwaGljcy5TaGFwZXM7CnVzaW5nIG9zdS5GcmFtZXdvcmsuR3JhcGhpY3MuVXNlckludGVyZmFjZTsKdXNpbmcgb3N1LkZyYW1ld29yay5Mb2NhbGlzYXRpb247CnVzaW5nIG9zdVRLOwp1c2luZyBvc3VUSy5HcmFwaGljczsK")
src = pp_blob.decode("utf-8")
print("=== PowerProgress.cs key fields ===")
for line in src.split("\n"):
    for kw in ["ProgressPerBar", "PowerRequiredFor", "requiredNormal", "requiredSuper", "smallBars", "largeBars"]:
        if kw in line:
            print(" ", line.rstrip())
            break

# Now find PlayerInfo.cs in the tree
raw2 = open(r"C:\Users\phili\.cursor\projects\c-Users-phili-AWBW\agent-tools\9acda395-f9fa-4f32-b33d-f34e8693e840.txt").read()
# Find PlayerInfo, COInfo
for fname in ["PlayerInfo", "COInfo", "COData", "ReplayController"]:
    m = re.search(rf'"path":\s*"([^"]*{fname}[^"]*\.cs)"', raw2)
    if m:
        path = m.group(1)
        sha_m = re.search(r'"sha":\s*"([a-f0-9]+)"', raw2[raw2.find(f'"{path}"'):raw2.find(f'"{path}"')+300])
        sha = sha_m.group(1) if sha_m else "?"
        print(f"\n{fname}: {path}  sha={sha}")
