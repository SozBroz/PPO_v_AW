#!/usr/bin/env python3
"""Probe: replay 1622501 via desync_audit's wrapper and trace Rachel SCOP."""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import game as gm

orig = gm.GameState._apply_power_effects
fired = {"count": 0}

def traced(self, player, cop):
    co = self.co_states[player]
    if co.co_id == 28 and not cop:
        aoe = self._oracle_power_aoe_positions
        ctype = type(aoe).__name__
        klen = len(aoe) if aoe else 0
        opp = 1 - player
        opp_units = self.units[opp]
        if isinstance(aoe, Counter):
            hits = sum(aoe.get(u.pos, 0) for u in opp_units)
        else:
            hits = -1
        print(f"  [trace] Rachel SCOP fires: aoe={ctype} keys={klen} opp_units={len(opp_units)} matched_hits={hits}")
        fired["count"] += 1
    orig(self, player, cop)

gm.GameState._apply_power_effects = traced

# Now run desync_audit on the gid
sys.argv = ["desync_audit", "--games-id", "1622501", "--register", "logs/_rachel_probe.jsonl"]
import tools.desync_audit as da
da.main()
print(f"\nrachel_scop_fires_total={fired['count']}")
