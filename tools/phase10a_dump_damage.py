"""Dump damage table rows for relevant attackers."""
import json
from engine.unit import UnitType, UNIT_STATS

table = json.load(open("data/damage_table.json", encoding="utf-8"))["table"]
for atk in [
    UnitType.RECON, UnitType.B_COPTER, UnitType.MECH, UnitType.MEGA_TANK,
    UnitType.MED_TANK, UnitType.NEO_TANK, UnitType.TANK, UnitType.BLACK_BOAT,
    UnitType.GUNBOAT, UnitType.CRUISER, UnitType.SUBMARINE,
]:
    a = int(atk)
    print(f"\n{atk.name} (max_ammo={UNIT_STATS[atk].max_ammo}, class={UNIT_STATS[atk].unit_class}):")
    for d in UnitType:
        v = table[a][int(d)]
        marker = " "
        print(f"  vs {d.name:12s}: {v}")
