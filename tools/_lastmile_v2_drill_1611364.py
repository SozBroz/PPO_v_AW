"""Quick drill of gid 1611364: P1 funds engine=3500 php=700 (delta +2800) at env 21 End."""
import json
import zipfile

zp = 'replays/amarriner_gl/1611364.zip'
zf = zipfile.ZipFile(zp)
data = zf.read('a1611364').decode('utf-8', errors='replace')
frames = []
for ln in data.splitlines():
    if not ln.strip():
        continue
    try:
        frames.append(json.loads(ln))
    except Exception:
        continue
print(f"total frames: {len(frames)}")
for fi in range(max(0, len(frames) - 6), len(frames)):
    f = frames[fi]
    ps = f.get('players', {})
    if isinstance(ps, dict):
        ps = list(ps.values())
    rich = [(p.get('id'), p.get('funds'), p.get('co_id')) for p in ps]
    print(f"frame {fi} day={f.get('day')} players_count={len(ps)} funds={rich}")
print()
print("Last 3 PHP envelopes (action stream, file '1611364'):")
acts_data = zf.read('1611364').decode('utf-8', errors='replace')
acts_lines = [ln for ln in acts_data.splitlines() if ln.strip()]
print(f"action lines: {len(acts_lines)}")
# print last 5 action lines
for i, ln in enumerate(acts_lines[-10:], start=len(acts_lines) - 10):
    try:
        obj = json.loads(ln)
        kind = obj.get('action') if isinstance(obj, dict) else None
        print(f"  line {i} kind={kind}: {ln[:300]}")
    except Exception:
        print(f"  line {i} (parse fail): {ln[:200]}")
