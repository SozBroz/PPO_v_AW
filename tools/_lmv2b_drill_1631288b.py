"""Count income-producing properties per seat at each frame for 1631288."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay

INCOME_TERRAIN_IDS = {34, 35, 36, 37, 38, 39, 40, 41, 42, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117}
# AWBW terrain ids: 34=neutral city, 35-41=country city. Let me just print all and see.

zp = Path('replays/amarriner_gl/1631288.zip')
frames = load_replay(zp)
print(f"frames: {len(frames)}")

terrain_ids_seen = set()
for f in frames[:1]:
    bldgs = f.get('buildings') or {}
    if isinstance(bldgs, dict):
        bldgs = list(bldgs.values())
    for b in bldgs:
        if isinstance(b, dict):
            terrain_ids_seen.add(b.get('terrain_id'))

print(f"terrain_ids in buildings: {sorted([t for t in terrain_ids_seen if t is not None])}")

# Count buildings per owner per frame
for fi in range(min(12, len(frames))):
    f = frames[fi]
    bldgs = f.get('buildings') or {}
    if isinstance(bldgs, dict):
        bldgs = list(bldgs.values())
    owner_terrain = defaultdict(lambda: defaultdict(int))
    for b in bldgs:
        if isinstance(b, dict):
            owner_terrain[b.get('players_id')][b.get('terrain_id')] += 1
    pretty = {k: dict(v) for k, v in owner_terrain.items()}
    ps = f.get('players', {})
    if isinstance(ps, dict):
        ps = list(ps.values())
    funds = {p.get('id'): p.get('funds') for p in ps if isinstance(p, dict)}
    print(f"\nframe {fi:2} day={f.get('day'):3} funds={funds}")
    for owner, terrains in pretty.items():
        print(f"  owner={owner}: {terrains} (total={sum(terrains.values())})")
