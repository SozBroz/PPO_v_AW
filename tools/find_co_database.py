"""Find CO database files in the viewer assets and check valid co_ids."""
import os, json
from pathlib import Path

native = Path(r"C:\Users\phili\AWBW\tools\awbw-player\lib\native")

# Look for json/ini files mentioning CO ids or CO data
found_files = []
for root, dirs, files in os.walk(native):
    for f in files:
        if f.lower().endswith((".json", ".ini", ".txt", ".csv")):
            found_files.append(os.path.join(root, f))

print(f"Config/data files: {len(found_files)}")
for fp in sorted(found_files)[:30]:
    print(f"  {fp}")

# Look for "CO" related folders
print("\nFolder names:")
for root, dirs, files in os.walk(native):
    for d in dirs:
        if "co" in d.lower() or "commander" in d.lower():
            print(f"  {os.path.join(root, d)}")
    break  # only top level

# Check if there's a data subfolder
data_dir = native / "data"
if data_dir.exists():
    for f in data_dir.iterdir():
        print(f"  data/{f.name}")
