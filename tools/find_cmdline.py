"""Check if the viewer accepts command-line args for file paths."""
import re

raw = open(r"C:\Users\phili\.cursor\projects\c-Users-phili-AWBW\agent-tools\9acda395-f9fa-4f32-b33d-f34e8693e840.txt").read()
# Find Program.cs or main entry point
for name in ["Program.cs", "AWBWApp.Desktop.cs", "GameBase.cs", "AWBWAppGameBase.cs"]:
    m = re.search(rf'"path":\s*"([^"]*{re.escape(name)}[^"]*)"', raw)
    if m:
        path = m.group(1)
        print(f"Found: {path}")
        # Get sha
        sha_m = re.search(r'"sha":\s*"([a-f0-9]+)"', raw[raw.find(f'"{path}"'):raw.find(f'"{path}"')+300])
        sha = sha_m.group(1) if sha_m else "?"
        print(f"  sha: {sha}")
