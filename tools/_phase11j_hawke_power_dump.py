"""Inspect Hawke power activations across the 3 residual zips.

Phase 11J-FINAL-HAWKE-CLUSTER-OWNER recon helper.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip  # type: ignore


def main() -> int:
    targets = sys.argv[1:] or [
        "replays/amarriner_gl/1617442.zip",
        "replays/amarriner_gl/1635679.zip",
        "replays/amarriner_gl/1635846.zip",
    ]
    for raw in targets:
        path = REPO / raw
        print(f"\n=== {path.name} ===")
        if not path.exists():
            print("  MISSING")
            continue
        envs = parse_p_envelopes_from_zip(path)
        for env_idx, (pid, day, actions) in enumerate(envs):
            for ai, a in enumerate(actions):
                if not isinstance(a, dict) or a.get("action") != "Power":
                    continue
                co_name = a.get("coName")
                power_name = a.get("powerName")
                co_power = a.get("coPower")
                ur = a.get("unitReplace")
                if isinstance(ur, str):
                    try:
                        ur = json.loads(ur)
                    except Exception:
                        ur = None
                own_post = Counter()
                enemy_post = Counter()
                if isinstance(ur, list):
                    for entry in ur:
                        if not isinstance(entry, dict):
                            continue
                        u_pid = entry.get("units_players_id") or entry.get("playerId")
                        hit = entry.get("units_hit_points")
                        if hit is None:
                            continue
                        try:
                            hit_i = int(hit)
                        except (TypeError, ValueError):
                            continue
                        if u_pid == pid:
                            own_post[hit_i] += 1
                        else:
                            enemy_post[hit_i] += 1
                print(
                    f"  env[{env_idx:>2}] day={day} pid={pid} co={co_name!r} "
                    f"power={power_name!r} flag={co_power!r}\n"
                    f"      own_post_hp={dict(sorted(own_post.items()))} "
                    f"(n_own={sum(own_post.values())})\n"
                    f"      enemy_post_hp={dict(sorted(enemy_post.items()))} "
                    f"(n_enemy={sum(enemy_post.values())})"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
