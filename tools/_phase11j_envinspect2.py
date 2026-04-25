#!/usr/bin/env python3
"""Phase 11J — full dump of failing Fire blocks (with full defender combatInfoVision)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip  # noqa: E402

# (gid, env_idx, action_idx_in_env)
TARGETS = [
    (1622104, 43, 19),
    (1625784, 35, 28),
    (1630983, 24, 1),
    (1631494, 46, 10),
    (1634664, 23, 3),
    (1635025, 36, 21),
    (1635846, 31, 15),
]

def main() -> int:
    out = []
    for gid, env_idx, fail_j in TARGETS:
        zpath = ROOT / "replays" / "amarriner_gl" / f"{gid}.zip"
        envs = parse_p_envelopes_from_zip(zpath)
        pid, day, actions = envs[env_idx]
        if fail_j is None:
            # find last Fire action
            fires = [j for j, a in enumerate(actions) if a.get("action") == "Fire"]
            fail_j = fires[-1] if fires else 0
        obj = actions[fail_j]
        if obj.get("action") != "Fire":
            # find next Fire after fail_j
            for j in range(fail_j, len(actions)):
                if actions[j].get("action") == "Fire":
                    fail_j = j
                    obj = actions[j]
                    break
        # extract full Fire
        fire = obj.get("Fire") or {}
        move = obj.get("Move") or {}
        out.append({
            "gid": gid, "env_idx": env_idx, "fail_j": fail_j,
            "envelope_pid": pid, "day": day,
            "Fire": fire, "Move": move,
        })
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
