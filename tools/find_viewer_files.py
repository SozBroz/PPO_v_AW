"""Find all relevant viewer source files from the cached repo tree JSON."""
import re

raw = open(r"C:\Users\phili\.cursor\projects\c-Users-phili-AWBW\agent-tools\9acda395-f9fa-4f32-b33d-f34e8693e840.txt").read()

# Extract (sha, path) pairs — the JSON has "path" then "sha" in each object
entries = re.findall(r'"path":\s*"([^"]+)",\s*"mode":[^,]+,\s*"type":\s*"blob",\s*"sha":\s*"([a-f0-9]{40})"', raw)

# Filter interesting ones
keywords = ['map', 'grid', 'terrain', 'building', 'replay', 'parser', 'awbwgame', 'action']
print(f"Total blobs: {len(entries)}\n")
for path, sha in entries:
    if not path.endswith('.cs'):
        continue
    low = path.lower()
    if any(k in low for k in keywords):
        print(f"{sha}  {path}")
