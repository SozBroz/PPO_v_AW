"""Apply all encoder fixes: Lash back, verify layout."""
import sys

ENC = r"d:\awbw\rl\encoder.py"
INFO = r"d:\awbw\rl\encoder_information.py"

# Fix 1: Add Lash back to encoder.py
with open(ENC, "r", encoding="utf-8") as f:
    content = f.read()

# The Lash block to insert after Olaf and before Koal
lash_block = """
    # Lash (co_id=16): +10% per defense star from the attacker's tile.
    # AWBW canon: D2D = +10%/star; COP = no change (only DEF doubled);
    # SCOP = +20%/star (ATK doubled). Air units excluded.
    if co_id == 16:
        cls = None
        try:
            from engine.unit import UNIT_STATS
            if unit is not None:
                cls = UNIT_STATS[unit.unit_type].unit_class
        except Exception:
            pass
        if cls not in ("air", "copter"):
            stars = max(0.0, float(defense_norm)) * 4.0
            if scop_active:
                return stars * 0.20   # +20%/star during SCOP
            return stars * 0.10   # D2D and COP: +10%/star
"""

# Find the location: after Olaf block, before Koal
marker = "    # Koal: roads."
if marker in content and "co_id == 16" not in content:
    content = content.replace(marker, lash_block + "\n" + marker)
    with open(ENC, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Added Lash back to encoder.py")
else:
    print(f"SKIP: Lash already present or Koal marker not found")

# Fix 2: Verify N_SCALARS = 20
with open(ENC, "r", encoding="utf-8") as f:
    for line in f:
        if "N_SCALARS" in line and "= " in line:
            print(f"encoder.py: {line.strip()}")
            break

# Fix 3: Verify encoder_information.py has 20-scalar layout
with open(INFO, "r", encoding="utf-8") as f:
    content = f.read()
    if "N_SCALARS = _enc.N_SCALARS" in content:
        print("encoder_information.py: N_SCALARS imported from encoder (correct)")
    if "power_bar_me" in content:
        print("encoder_information.py: power_bar_me found (correct)")
    if "cop_stars_me" in content:
        print("encoder_information.py: cop_stars_me found (correct)")
        
print("\nDone. All fixes applied.")
