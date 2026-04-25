"""Probe gid 1632047 — find PHP unit 192597060 across frames."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.diff_replay_zips import load_replay

ZIP = Path("replays/amarriner_gl/1632047.zip")
TARGET_ID = 192597060

frames = load_replay(ZIP)
print(f"frames: {len(frames)}")
for i, snap in enumerate(frames):
    units = snap.get("units") or {}
    if isinstance(units, dict):
        ulist = list(units.values())
    else:
        ulist = units
    found = None
    for u in ulist:
        if isinstance(u, dict):
            try:
                if int(u.get("id", -1)) == TARGET_ID:
                    found = u
                    break
            except (TypeError, ValueError):
                pass
    if found:
        print(f"  snap[{i}] day={snap.get('day')} turn={snap.get('turn')} unit={found.get('id')} pos=({found.get('x')},{found.get('y')}) hp={found.get('hit_points')} pid={found.get('players_id')} type={found.get('units_name')}")
    elif i in (22, 23, 24):
        print(f"  snap[{i}] day={snap.get('day')} turn={snap.get('turn')} unit MISSING (looked for id {TARGET_ID})")
