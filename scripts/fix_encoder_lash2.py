"""Fix encoder.py: add Lash back to _co_tile_attack_bonus_for_category."""
# Read the file
with open(r"d:\awbw\rl\encoder.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find where to insert Lash (after Olaf block, before Koal)
output = []
$i = 0
while i < len(lines):
    line = lines[i]
    output.append(line)
    
    # Check if this is the Olaf return statement followed by blank line and "# Koal:"
    if ('# Olaf (co_id=9)' in line or 
        (i > 0 and '# Olaf' in lines[i-1] and 'return 0.0' in line)):
        # Look ahead to see if next non-empty line is "# Koal:"
        j = i + 1
        while j < len(lines) and lines[j].strip() == '':
            j += 1
        if j < len(lines) and '# Koal:' in lines[j]:
            # Insert Lash block before Koal
            output.append("\n")
            output.append("    # Lash (co_id=16): +10% per defense star from the attacker's tile.\n")
            output.append("    # AWBW canon: D2D = +10%/star; COP = no change (only DEF doubled);\n")
            output.append("    # SCOP = +20%/star (ATK doubled). Air units excluded.\n")
            output.append("    if co_id == 16:\n")
            output.append("        if category in (\"air\", \"copter\"):\n")
            output.append("            return 0.0\n")
            output.append("        stars = max(0.0, float(defense_norm)) * 4.0\n")
            output.append("        if scop_active:\n")
            output.append("            return stars * 0.20   # +20%/star during SCOP\n")
            output.append("        # D2D: +10%/star (COP only doubles defense, not ATK)\n")
            output.append("        return stars * 0.10\n")
            output.append("\n")
    i += 1

# Write back
with open(r"d:\awbw\rl\encoder.py", "w", encoding="utf-8") as f:
    f.writelines(output)

print("SUCCESS: Added Lash back to _co_tile_attack_bonus_for_category")
print("Verifying...")
with open(r"d:\awbw\rl\encoder.py", "r", encoding="utf-8") as f:
    content = f.read()
if "Lash (co_id=16)" in content:
    print("VERIFIED: Lash section is present")
else:
    print("WARNING: Lash section not found")
