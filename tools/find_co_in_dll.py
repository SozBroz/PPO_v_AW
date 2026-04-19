"""
Extract CO IDs and names from the AWBWApp.Resources.dll (or Game.dll).
Looking for patterns like: {10, "Kindle"} or "Kindle" near integer 10.
Also check what CO IDs are actually in the AWBW DB.
"""
import re
from pathlib import Path

native = Path(r"C:\Users\phili\AWBW\tools\awbw-player\lib\native")

# Check which DLLs contain "CommandingOfficer" or "CO" data
for dll_path in sorted(native.glob("AWBWApp*.dll")):
    data = open(dll_path, "rb").read()
    # Look for CO names in utf-16-le
    co_names = ["Nell", "Andy", "Sami", "Max", "Grit", "Sturm", "Eagle", "Drake",
                "Kanbei", "Sonja", "Hachi", "Hawke", "Colin", "Sasha", "Sensei",
                "Grimm", "Kindle", "Jugger", "Koal", "Lash", "Jake", "Rachel",
                "Von Bolt", "Flak", "Adder", "Clone Andy"]
    hits = []
    for name in co_names:
        n16 = name.encode("utf-16-le")
        if n16 in data:
            idx = data.index(n16)
            hits.append(name)
    if hits:
        print(f"\n{dll_path.name}: found COs {hits[:10]}")
        # Now try to extract co_id -> name mapping by looking at surrounding bytes
        for name in ["Nell", "Andy", "Sami", "Kindle", "Jugger"]:
            n16 = name.encode("utf-16-le")
            pos = data.find(n16)
            if pos < 0:
                continue
            # Look for an integer nearby (within 32 bytes before)
            nearby = data[max(0, pos-32):pos]
            # Find 4-byte little-endian int values
            ints = []
            for i in range(0, len(nearby)-3):
                val = int.from_bytes(nearby[i:i+4], "little")
                if 0 < val < 100:
                    ints.append((i, val))
            print(f"  '{name}' @ {pos:#x}, nearby ints: {ints}")
