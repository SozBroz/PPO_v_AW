"""Dump full content of Move and Build actions in env 4."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.desync_audit import parse_p_envelopes_from_zip

zp = Path('replays/amarriner_gl/1631288.zip')
envelopes = parse_p_envelopes_from_zip(zp)
env4 = envelopes[4]
pid, day, actions = env4
for ai in (2,):
    a = actions[ai]
    print(f"=== action {ai} kind={a.get('action')} ===")
    print(json.dumps(a, default=str, indent=2)[:3000])
    print()
