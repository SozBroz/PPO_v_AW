"""Verify observer-relative encoding swaps correctly."""
import sys
sys.path.insert(0, r"d:\awbw")

from rl import encoder as enc
import numpy as np

# Quick verification of observer-relative encoding
print("=== Verifying observer-relative encoding ===")
print()

# 1. Check that _fill_co_tile_attack_bonus_planes uses (observer, enemy) order
print("1. CO tile attack bonus planes:")
with open(r"d:\awbw\rl\encoder.py", "r") as f:
    content = f.read()
idx = content.find("_fill_co_tile_attack_bonus_planes")
if idx >= 0:
    snippet = content[idx:idx+800]
    if "seats = (int(observer), 1 - int(observer))" in snippet:
        print("   OK: seats are (observer, enemy)")
    else:
        print("   CHECK: seats definition not found")
        # Show what's there
        for line in snippet.split("\n")[:20]:
            if "seat" in line.lower() or "observer" in line.lower():
                print(f"   Found: {line.strip()}")

# 2. Check scalar encoding uses observer/enemy correctly  
print("\n2. Scalar encoding:")
if "co_me = state.co_states[observer]" in content:
    print("   OK: co_me = co_states[observer]")
if "co_en = state.co_states[enemy]" in content:
    print("   OK: co_en = co_states[enemy] (where enemy = 1 - observer)")
if "enemy = 1 - int(observer)" in content:
    print("   OK: enemy = 1 - observer")

# 3. Check that me/enemy funds are in correct position
print("\n3. Scalar layout verification:")
if "scalars[0] = state.funds[observer]" in content:
    print("   OK: scalars[0] = funds[observer] (me)")
if "scalars[1] = state.funds[enemy]" in content:
    print("   OK: scalars[1] = funds[enemy]")

# 4. Check power_bar encoding
print("\n4. Power bar encoding:")
if "scalars[2] = co_me.power_bar" in content:
    print("   OK: scalars[2] = co_me.power_bar (me)")
if "scalars[7] = co_en.power_bar" in content:
    print("   OK: scalars[7] = co_en.power_bar (enemy)")

# 5. Summary
print("\n=== Summary ===")
print(f"N_SCALARS = {enc.N_SCALARS}")
print(f"N_SPATIAL_CHANNELS = {enc.N_SPATIAL_CHANNELS}")
print(f"Scalar labels: {len(enc.SCALAR_LABELS)} entries")
for i, label in enumerate(enc.SCALAR_LABELS):
    print(f"   [{i:2d}] {label}")
