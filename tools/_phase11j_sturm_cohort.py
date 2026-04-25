"""Phase 11J-FINAL Sturm cohort scan.

Scans replays/amarriner_gl/*.zip for any Power envelope where coName == 'Sturm'.
Lists gid, env, day, kind (S/Y), powerName, missileCoords, #affected units.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip


def scan_zip(zpath: Path):
    try:
        envs = parse_p_envelopes_from_zip(zpath)
    except Exception as e:
        return ("error", str(e))
    out = []
    for i, (pid, day, acts) in enumerate(envs):
        for a in acts:
            if isinstance(a, dict) and a.get("action") == "Power" and a.get("coName") == "Sturm":
                mc = a.get("missileCoords")
                ur = a.get("unitReplace") or {}
                glob = ur.get("global", {}) if isinstance(ur, dict) else {}
                us = glob.get("units") if isinstance(glob, dict) else None
                n_affected = len(us) if isinstance(us, list) else 0
                out.append({
                    "env": i, "pid": pid, "day": day,
                    "kind": a.get("coPower"),
                    "name": a.get("powerName"),
                    "missileCoords": mc,
                    "n_affected": n_affected,
                })
    return out


def main():
    zdir = Path("replays/amarriner_gl")
    zips = sorted(zdir.glob("*.zip"))
    sturm = []
    for zp in zips:
        try:
            gid = int(zp.stem)
        except ValueError:
            continue
        result = scan_zip(zp)
        if isinstance(result, tuple):
            continue
        if result:
            sturm.append((gid, result))
    print(f"=== {len(sturm)} zips with Sturm Power envelopes (out of {len(zips)} total) ===")
    for gid, powers in sturm:
        for p in powers:
            print(f"gid {gid} env {p['env']} day {p['day']} kind={p['kind']} name={p['name']!r} mc={p['missileCoords']} n={p['n_affected']}")


if __name__ == "__main__":
    main()
