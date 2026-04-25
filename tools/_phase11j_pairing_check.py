from pathlib import Path
import sys
sys.path.insert(0, '.')
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

for gid in (1621434, 1621898, 1622328, 1624082):
    zp = Path(f'replays/amarriner_gl/{gid}.zip')
    envs = parse_p_envelopes_from_zip(zp)
    frames = load_replay(zp)
    if len(frames) == len(envs) + 1:
        mode = 'trailing'
    elif len(frames) == len(envs):
        mode = 'tight'
    else:
        mode = 'other'
    print(f'gid={gid}: n_envelopes={len(envs)}, n_frames={len(frames)}, mode={mode}')
