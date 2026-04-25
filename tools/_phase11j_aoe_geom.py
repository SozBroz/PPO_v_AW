"""Verify which AOE shape matches PHP's unitReplace for a Rachel SCOP envelope."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay


def main(gid: int, env_idx: int) -> None:
    f = load_replay(Path(f"replays/amarriner_gl/{gid}.zip"))
    pre_frame = f[env_idx]

    from tools.oracle_zip_replay import parse_p_envelopes_from_zip
    envs = parse_p_envelopes_from_zip(Path(f"replays/amarriner_gl/{gid}.zip"))
    pid, day, actions = envs[env_idx]
    power = None
    for a in actions:
        if a.get("action") == "Power":
            power = a; break
    assert power is not None
    centers = []
    for entry in power.get("missileCoords") or []:
        centers.append((int(entry["x"]), int(entry["y"])))

    affected_ids = set()
    affected_post = {}
    for u in (power.get("unitReplace") or {}).get("global", {}).get("units") or []:
        affected_ids.add(int(u["units_id"]))
        affected_post[int(u["units_id"])] = u.get("units_hit_points")

    by_id_pre = {}
    for u in (pre_frame.get("units") or {}).values():
        try:
            uid = int(u.get("id"))
        except (TypeError, ValueError):
            continue
        by_id_pre[uid] = u

    print(f"=== gid {gid} env {env_idx} (day {day}) Rachel SCOP ===")
    print(f"  centers: {centers}")
    print()
    print(f"  PHP unitReplace (n={len(affected_ids)}):")
    for uid in affected_ids:
        u = by_id_pre.get(uid)
        if u is None:
            print(f"    id={uid} <not in pre-frame>"); continue
        ux, uy = int(u["x"]), int(u["y"])
        try:
            pre_hp = float(u.get("hit_points"))
        except (TypeError, ValueError):
            pre_hp = None
        post_hp = affected_post[uid]
        dists = [abs(ux - cx) + abs(uy - cy) for (cx, cy) in centers]
        # Count hits at each Manhattan threshold
        hits1 = sum(1 for d in dists if d <= 1)
        hits2 = sum(1 for d in dists if d <= 2)
        hits_box1 = sum(1 for d, (cx, cy) in zip(dists, centers)
                        if max(abs(ux - cx), abs(uy - cy)) <= 1)
        try:
            dmg = (float(pre_hp) - float(post_hp)) if pre_hp is not None else None
        except (TypeError, ValueError):
            dmg = None
        print(f"    id={uid} {u.get('name'):14s} pid={u.get('players_id')} pos=({ux},{uy}) "
              f"hp {pre_hp}->{post_hp} dmg={dmg}  "
              f"min_manhattan={min(dists)}  hits(M<=1)={hits1}  hits(M<=2)={hits2}  hits(box<=1)={hits_box1}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
