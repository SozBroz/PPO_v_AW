#!/usr/bin/env python3
"""Phase 11J-CLUSTER-B-SHIP — find Von Bolt SCOP envelopes in target gids."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

ZIPS = ROOT / "replays" / "amarriner_gl"

for gid in (1622328, 1623698, 1629521):
    zpath = ZIPS / f"{gid}.zip"
    if not zpath.exists():
        print(f"gid={gid} MISSING")
        continue
    envs = parse_p_envelopes_from_zip(zpath)
    for ei, (pid, day, acts) in enumerate(envs):
        for j, a in enumerate(acts):
            if not isinstance(a, dict):
                continue
            if a.get("action") != "Power":
                continue
            cn = a.get("coName")
            cop = a.get("coPower")
            mc = a.get("missileCoords")
            print(f"gid={gid} env={ei:>3} idx={j:>3} day={day:>3} pid={pid} co={cn} pwr={cop} missile={mc}")
