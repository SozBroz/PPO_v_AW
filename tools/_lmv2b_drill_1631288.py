"""Drill gid 1631288 (Sami P0 vs Sonja P1) funds delta at env 7 End (day 4 -> day 5)."""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay

zp = Path('replays/amarriner_gl/1631288.zip')
frames = load_replay(zp)
print(f"frames: {len(frames)}")

def summarize(fi, f):
    ps = f.get('players', {})
    if isinstance(ps, dict):
        ps = list(ps.values())
    funds = {p.get('id'): (p.get('funds'), p.get('co_id')) for p in ps if isinstance(p, dict)}
    units = f.get('units') or {}
    if isinstance(units, dict):
        units = list(units.values())
    by_seat = defaultdict(int)
    for u in units:
        if isinstance(u, dict):
            by_seat[u.get('players_id')] += 1
    bldgs = f.get('buildings') or {}
    if isinstance(bldgs, dict):
        bldgs = list(bldgs.values())
    by_owner = defaultdict(int)
    for b in bldgs:
        if isinstance(b, dict):
            by_owner[b.get('players_id')] += 1
    print(f"frame {fi:3} day={f.get('day'):3} player={f.get('current_turn_pid', '?')} funds={funds} units={dict(by_seat)} props={dict(by_owner)}")

for fi in range(min(12, len(frames))):
    summarize(fi, frames[fi])

# Diff env 7 -> env 8 (frame 7 -> frame 8)
if len(frames) > 8:
    f7, f8 = frames[7], frames[8]
    bldgs7 = f7.get('buildings') or {}
    if isinstance(bldgs7, dict):
        bldgs7 = list(bldgs7.values())
    bldgs8 = f8.get('buildings') or {}
    if isinstance(bldgs8, dict):
        bldgs8 = list(bldgs8.values())
    keyfn = lambda b: (b.get('x'), b.get('y'))
    map7 = {keyfn(b): b for b in bldgs7 if isinstance(b, dict)}
    map8 = {keyfn(b): b for b in bldgs8 if isinstance(b, dict)}
    print("\n--- Building diffs frame 7 -> 8 (envelope 7 = day 4 P1 End) ---")
    for k in sorted(set(map7) | set(map8), key=lambda kk: (kk[1] or 0, kk[0] or 0)):
        b7 = map7.get(k)
        b8 = map8.get(k)
        if b7 is None or b8 is None:
            print(f"  pos={k} only_in={'7' if b8 is None else '8'}")
            continue
        diffs = []
        for kk in ('players_id', 'last_capture', 'capture', 'terrain_id', 'last_capture_pid'):
            v7 = b7.get(kk)
            v8 = b8.get(kk)
            if v7 != v8:
                diffs.append(f"{kk}:{v7}->{v8}")
        if diffs:
            print(f"  pos={k} terrain={b7.get('terrain_id')} {' | '.join(diffs)}")
