"""Dump PHP unit HPs around the env 26/27 transition for gid 1621434."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

GID = 1621434
zp = Path(f'replays/amarriner_gl/{GID}.zip')
envs = parse_p_envelopes_from_zip(zp)
frames = load_replay(zp)
print(f'n_envelopes={len(envs)}, n_frames={len(frames)}')

# Frame map for env i: frame[i+1] is the post-envelope frame (per drill).
# But tight mode: n_frames == n_envelopes, so frame[i] is paired to env i-1 transitions.
# Phase 10N drill uses frame[step_i + 1] when exists.

target_envs = [27]
# Print frame[27] (= pre-env-27 = post-env-26 in tight) and frame[28] (post-env-27).
for ei in target_envs:
    print(f'\n=== env {ei} = {envs[ei][0]} day={envs[ei][1]} ===')
    for fi in (ei, ei+1):
        if fi >= len(frames):
            print(f' (no frame {fi})')
            continue
        f = frames[fi]
        print(f' frame[{fi}]:')
        print(f'   day={f.get("day")} active_player_id={f.get("active_player_id")} turn={f.get("turn")}')
        # Funds
        for k, p in (f.get('players') or {}).items():
            print(f'   player {p.get("id")} funds={p.get("funds")} co={p.get("co_id")}')
        # Show only P0 units (3747996), sorted by pos
        rows = []
        for k, u in (f.get('units') or {}).items():
            if str(u.get('players_id')) != '3747996':
                continue
            hp = u.get('hit_points')
            x = u.get('units_x')
            y = u.get('units_y')
            ut = u.get('units_name')
            rows.append(((x or 0, y or 0), ut, hp))
        rows.sort()
        for pos, ut, hp in rows:
            print(f'   P0 {ut} pos={pos} hp={hp}')
