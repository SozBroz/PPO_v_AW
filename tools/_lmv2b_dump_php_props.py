"""Dump PHP frame 7 building dict structure to find ownership encoding."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay

zp = Path('replays/amarriner_gl/1631288.zip')
frames = load_replay(zp)
f7 = frames[7]
bldgs7 = f7.get('buildings') or {}
if isinstance(bldgs7, dict):
    bldgs7 = list(bldgs7.values())

print(f"sample building keys: {sorted(bldgs7[0].keys()) if bldgs7 else None}")
print(f"first 3 buildings:")
for b in bldgs7[:3]:
    print(f"  {b}")
print()

# Find non-neutral terrain buildings (terrain >= 38)
non_neutral = [b for b in bldgs7 if isinstance(b, dict) and (b.get('terrain_id') or 0) >= 38 and (b.get('terrain_id') or 0) <= 116]
print(f"non-neutral player props in f7: {len(non_neutral)}")
# Group by terrain
from collections import Counter
print(Counter([b.get('terrain_id') for b in non_neutral]))

# Check explicit ownership keys
keys_seen = defaultdict(set)
for b in non_neutral:
    for k, v in b.items():
        if 'player' in k.lower() or 'owner' in k.lower() or 'pid' in k.lower():
            keys_seen[k].add(v)
print(f"\nplayer/owner-related keys in non-neutral props: {dict(keys_seen)}")

# Check players list to confirm ID encoding
print(f"\nplayers in f7:")
ps = f7.get('players')
if isinstance(ps, dict):
    ps = list(ps.values())
for p in ps:
    print(f"  {p}")
