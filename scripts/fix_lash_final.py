"""Fix Lash block in encoder.py - remove invalid unit reference."""
ENC = r"d:\awbw\rl\encoder.py"

with open(ENC, "r", encoding="utf-8") as f:
    content = f.read()

# The broken Lash block
old = """    # Lash (co_id=16): +10% per defense star from the attacker's tile.
    # AWBW canon: D2D = +10%/star; COP = no change (only DEF doubled);
    # SCOP = +20%/star (ATK doubled). 
    # Air units: exclusion is handled in ``combat.py`` / ``attack_value_for_unit()``.
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
            return stars * 0.10"""

# Corrected block - map-position feature, air exclusion handled in combat
new = """    # Lash (co_id=16): +10% per defense star from the attacker's tile.
    # AWBW canon: D2D = +10%/star; COP = no change (only DEF doubled);
    # SCOP = +20%/star (ATK doubled). 
    # Air/copter exclusion is handled in ``combat.py`` / ``attack_value_for_unit()``.
    # This is a map-position feature - the NN learns air units don't benefit.
    if co_id == 16:
        stars = max(0.0, float(defense_norm)) * 4.0
        if scop_active:
            return stars * 0.20   # +20%/star during SCOP
        # D2D and COP: +10%/star (COP only doubles defense, not ATK)
        return stars * 0.10"""

if old in content:
    content = content.replace(old, new)
    with open(ENC, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Fixed Lash block - removed invalid 'unit' reference")
else:
    print("ERROR: Old block not found")
    # Debug: show what's around Lash
    idx = content.find("Lash (co_id=16)")
    if idx >= 0:
        print(f"Found Lash at index {idx}")
        # Show the actual text
        snippet = content[idx:idx+600]
        print(repr(snippet))
