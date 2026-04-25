#!/usr/bin/env python3
"""Cross-ref every Delete in failing envelopes to PHP unit position."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import parse_p_envelopes_from_zip

ZIPS = ROOT / "replays" / "amarriner_gl"


def php_unit_pos(frame, units_id):
    units = (frame.get("units") or {})
    for u in units.values():
        try:
            if int(u.get("units_id") or u.get("id")) == int(units_id):
                return (int(u["y"]), int(u["x"])), u.get("name"), u.get("players_id")
        except (TypeError, ValueError):
            continue
    return None, None, None


def main():
    for line in open(sys.argv[1], encoding="utf-8"):
        gid = int(line.strip())
        zpath = ZIPS / f"{gid}.zip"
        envs = parse_p_envelopes_from_zip(zpath)
        frames = load_replay(zpath)
        # Find the failing envelope by the register
        # Just iterate all envelopes and report any Delete + following Build
        for env_i, (pid, day, actions) in enumerate(envs):
            for j, obj in enumerate(actions):
                if not isinstance(obj, dict) or obj.get("action") != "Delete":
                    continue
                inner = obj.get("Delete") or {}
                uid_obj = inner.get("unitId") or {}
                uid = uid_obj.get("global") if isinstance(uid_obj, dict) else None
                if uid is None:
                    continue
                # Lookup pos in frames around this envelope.
                pos_pre, name_pre, pid_pre = php_unit_pos(frames[env_i], uid) if env_i < len(frames) else (None, None, None)
                pos_post, name_post, pid_post = php_unit_pos(frames[env_i + 1], uid) if env_i + 1 < len(frames) else (None, None, None)
                # Look ahead for next Build in this envelope
                next_build_pos = None
                next_build_dist = None
                for k in range(j + 1, len(actions)):
                    a = actions[k]
                    if isinstance(a, dict) and a.get("action") == "Build":
                        gu = a.get("unit") or a.get("newUnit") or {}
                        if "global" in gu:
                            gu = gu["global"]
                        try:
                            br = int(gu["units_y"])
                            bc = int(gu["units_x"])
                        except (KeyError, TypeError, ValueError):
                            br, bc = -1, -1
                        next_build_pos = (br, bc)
                        next_build_dist = k - j
                        break
                # Look back for prev Build (sometimes Delete follows Build for cleanup)
                prev_build_pos = None
                prev_build_dist = None
                for k in range(j - 1, -1, -1):
                    a = actions[k]
                    if isinstance(a, dict) and a.get("action") == "Build":
                        gu = a.get("unit") or a.get("newUnit") or {}
                        if "global" in gu:
                            gu = gu["global"]
                        try:
                            br = int(gu["units_y"])
                            bc = int(gu["units_x"])
                        except (KeyError, TypeError, ValueError):
                            br, bc = -1, -1
                        prev_build_pos = (br, bc)
                        prev_build_dist = j - k
                        break
                print(
                    f"gid={gid} env={env_i} day={day} pid={pid} act_idx={j} "
                    f"DEL_uid={uid} pre_pos={pos_pre} post_pos={pos_post} "
                    f"name={name_pre or name_post} owner_pid={pid_pre or pid_post} "
                    f"next_Build@dist={next_build_dist} pos={next_build_pos} "
                    f"prev_Build@dist={prev_build_dist} pos={prev_build_pos}"
                )


if __name__ == "__main__":
    main()
