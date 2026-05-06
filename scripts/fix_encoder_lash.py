"""Fix encoder.py: add Lash back to _co_tile_attack_bonus_for_category."""
# Read the file
with open(r"d:\awbw\rl\encoder.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find where to insert Lash (after Olaf section, before Koal)
output = []
$i = 0
while i < len(lines):
    line = lines[i]
    output.append(line)
    
    # After Olaf's return 0.0, add Lash section before Koal
    if '# Olaf (co_id=9)' in line and i + 8 < len(lines):
        # Check if next lines are Olaf block
        block = ''.join(lines[i:i+9])
        if 'return 0.0' in block and '# Koal:' in ''.join(lines[i+9:i+12]):
            # Insert Lash between Olaf and Koal
            output.append("\n")
            output.append("    # Lash (co_id=16): +10% per defense star from the attacker's tile.\n")
            output.append("    # AWBW canon: D2D = +10%/star; COP = no change (only DEF doubled);\n")
            output.append("    # SCOP = +20%/star (ATK doubled). Air units excluded.\n")
            output.append("    if co_id == 16:\n")
            output.append("        cls = None\n")
            output.append("        try:\n")
            output.append("            from engine.unit import UNIT_STATS\n")
            output.append("            if unit is not None:\n")
            output.append("                cls = UNIT_STATS[unit.unit_type].unit_class\n")
            output.append("        except Exception:\n")
            output.append("            pass\n")
            output.append("        if cls not in (\"air\", \"copter\"):\n")
            output.append("            stars = max(0.0, float(defense_norm)) * 4.0\n")
            output.append("            if scop_active:\n")
            output.append("                return stars * 0.20   # +20%/star during SCOP\n")
            output.append("            # D2D and COP: +10%/star (COP only doubles defense, not ATK)\n")
            output.append("            return stars * 0.10\n")
            output.append("\n")
            i += 9  # Skip past Olaf block
            continue
    i += 1

# Write back
with open(r"d:\awbw\rl\encoder.py", "w", encoding="utf-8") as f:
    f.writelines(output)

print("SUCCESS: Added Lash back to _co_tile_attack_bonus_for_category")
print("Verifying...")
# Verify
with open(r"d:\awbw\rl\encoder.py", "r", encoding="utf-8") as f:
    content = f.read()
    if "Lash (co_id=16)" in content:
        print("VERIFIED: Lash section is present")
    else:
        print("WARNING: Lash section not found")
