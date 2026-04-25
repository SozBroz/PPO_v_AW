"""Apply Phase 10A damage-table fixes to data/damage_table.json.

Adds AWBW-canonical entries that were null in the engine table and which
caused first-divergence engine_bug rows in the post-Phase-9 audit.

AWBW-canonical source: https://awbw.amarriner.com/damage.php
(B-Copter row + Recon row cross-checked).

Cells modified:
  B_COPTER (row 15) vs LANDER (col 21): None -> 25
  B_COPTER (row 15) vs BLACK_BOAT (col 23): None -> 25
  RECON    (row 2)  vs B_COPTER (col 15): None -> 10
  RECON    (row 2)  vs T_COPTER (col 16): None -> 35
"""
from __future__ import annotations

import json
from pathlib import Path

P = Path("data/damage_table.json")
data = json.loads(P.read_text(encoding="utf-8"))
table = data["table"]

cells = [
    (15, 21, 25, "B_COPTER vs LANDER"),
    (15, 23, 25, "B_COPTER vs BLACK_BOAT"),
    (2, 15, 10, "RECON vs B_COPTER"),
    (2, 16, 35, "RECON vs T_COPTER"),
]

for r, c, v, label in cells:
    old = table[r][c]
    table[r][c] = v
    print(f"  {label} [{r}][{c}]: {old} -> {v}")

new_note = (
    "2026-04 (Phase 10A): Filled B_COPTER vs LANDER/BLACK_BOAT (25/25) and "
    "RECON vs B_COPTER/T_COPTER (10/35) per AWBW canonical damage chart "
    "(https://awbw.amarriner.com/damage.php). Engine refused legitimate "
    "B-Copter strikes on adjacent landers/black boats and Recon MG fire "
    "on copters; 47 GL std-tier engine_bug rows in "
    "logs/desync_register_post_phase9.jsonl turn on these matchups."
)
notes = data.get("_notes") or []
notes.append(new_note)
data["_notes"] = notes

P.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\nWrote {P}")
