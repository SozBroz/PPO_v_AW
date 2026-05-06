"""Fix encoder.py: add Lash back to _co_tile_attack_bonus_for_category."""
import re

with open(r"d:\awbw\rl\encoder.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the exact location and add Lash between Olaf and Koal
old_olaf = """    # Olaf (co_id=9): +10% ATK/DEF in snow (D2D, AWBW amarriner page:
    # "Unaffected by snow, but rain affects him the same as snow would for others."
    # The hover tooltip in AWBW shows +10% ATK/DEF when snow is active for Olaf.
    if co_id == 9:
        weather = getattr(co_state, "_weather_cache", "clear")
        if weather == "snow":
            return 0.10  # terrain-independent +10% ATK/DEF
        return 0.0

    # Koal: roads."""

new_with_lash = """    # Olaf (co_id=9): +10% ATK/DEF in snow (D2D, AWBW amarriner page:
    # "Unaffected by snow, but rain affects him the same as snow would for others."
    # The hover tooltip in AWBW shows +10% ATK/DEF when snow is active for Olaf.
    if co_id == 9:
        weather = getattr(co_state, "_weather_cache", "clear")
        if weather == "snow":
            return 0.10  # terrain-independent +10% ATK/DEF
        return 0.0

    # Lash (co_id=16): +10% per defense star from the attacker's tile.
    # AWBW canon: D2D = +10%/star; COP = no change (only DEF doubled);
    # SCOP = +20%/star (ATK doubled). Air units excluded (canon:
    # "air units are unaffected by terrain").
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
            # D2D and COP: +10%/star (COP only doubles defense, not ATK)
            return stars * 0.10

    # Koal: roads."""

if old_olaf in content:
    content = content.replace(old_olaf, new_with_lash)
    with open(r"d:\awbw\rl\encoder.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Added Lash back to _co_tile_attack_bonus_for_category")
else:
    print("ERROR: Could not find the target string")
    # Try to show what's around that area
    idx = content.find("Olaf (co_id=9)")
    if idx >= 0:
        print(f"Found Olaf at index {idx}")
        print(repr(content[idx:idx+600]))
