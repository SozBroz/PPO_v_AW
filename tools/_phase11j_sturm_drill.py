"""Phase 11J-FINAL Sturm SCOP drill — list Power envelopes + missileCoords.

Usage: python tools/_phase11j_sturm_drill.py <gid>
"""
from __future__ import annotations
import json
import sys
import zipfile
import gzip
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip


def main(gid: int):
    zpath = Path(f"replays/amarriner_gl/{gid}.zip")
    envs = parse_p_envelopes_from_zip(zpath)
    print(f"=== gid {gid}: {len(envs)} envelopes ===")
    powers = []
    for i, (pid, day, acts) in enumerate(envs):
        for a in acts:
            if isinstance(a, dict) and a.get("action") == "Power":
                powers.append((i, pid, day, a))
    print(f"=== {len(powers)} Power envelopes ===")
    for i, pid, day, a in powers:
        co = a.get("coName")
        kind = a.get("coPower")
        name = a.get("powerName")
        mc = a.get("missileCoords")
        ur = a.get("unitReplace")
        print(f"\nenv {i} pid {pid} day {day} co={co} kind={kind} name={name!r}")
        if mc:
            print(f"  missileCoords: {mc}")
        if isinstance(ur, dict):
            glob = ur.get("global", {})
            us = glob.get("units") if isinstance(glob, dict) else None
            if isinstance(us, list):
                print(f"  unitReplace.global.units ({len(us)}):")
                for u in us:
                    print(f"    {u}")

    # Also dump pre-Power and post-Power snapshots if available from <gid> blob
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        snap_name = str(gid)
        if snap_name in names:
            raw = zf.read(snap_name)
            try:
                txt = gzip.decompress(raw).decode("utf-8", "replace")
            except Exception:
                txt = raw.decode("utf-8", "replace")
            print(f"\n=== snapshot blob {snap_name}: {len(txt)} chars ===")
            # PHP serialized; just count units
            n_units = txt.count('"units_id"')
            print(f"  rough units_id count: {n_units}")

if __name__ == "__main__":
    gid = int(sys.argv[1]) if len(sys.argv) > 1 else 1635679
    main(gid)
