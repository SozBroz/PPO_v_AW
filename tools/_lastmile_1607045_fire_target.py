"""Dump env 17 fires targeting our two affected Drake units."""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    parse_p_envelopes_from_zip,
    map_snapshot_player_ids_to_engine,
)


TARGET_PHP_IDS = {190277871, 190289865}


def main() -> int:
    gid = 1607045
    co0, co1 = 5, 28
    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)

    pid, day, actions = envs[17]
    print(f"env 17 day {day} pid={pid}")

    for j, obj in enumerate(actions):
        if obj.get("action") not in ("Fire", "AttackSeam"):
            continue
        ci = obj.get("combatInfo") or {}
        att = ci.get("attacker") if isinstance(ci, dict) else None
        defn = ci.get("defender") if isinstance(ci, dict) else None
        if not isinstance(defn, dict):
            continue
        try:
            d_uid = int(defn.get("units_id"))
        except Exception:
            continue
        if d_uid not in TARGET_PHP_IDS:
            continue
        a_uid = att.get("units_id") if isinstance(att, dict) else None
        a_hp = att.get("units_hit_points") if isinstance(att, dict) else None
        d_hp = defn.get("units_hit_points")
        a_name = att.get("units_name") if isinstance(att, dict) else None
        d_name = defn.get("units_name")
        a_pos = (att.get("units_y"), att.get("units_x")) if isinstance(att, dict) else None
        d_pos = (defn.get("units_y"), defn.get("units_x"))
        gv = obj.get("combatInfoVision") or {}
        # also get per-pid combatInfo
        for_per_pid_ks = list(gv.keys()) if isinstance(gv, dict) else []
        per_pid_ci = {}
        for k in for_per_pid_ks:
            v = gv.get(k)
            if isinstance(v, dict):
                inner = v.get("combatInfo") or v
                d2 = inner.get("defender") if isinstance(inner, dict) else None
                a2 = inner.get("attacker") if isinstance(inner, dict) else None
                per_pid_ci[k] = {
                    "att_hp": a2.get("units_hit_points") if isinstance(a2, dict) else None,
                    "def_hp": d2.get("units_hit_points") if isinstance(d2, dict) else None,
                }
        print(f"  [{j}] Fire att_id={a_uid} ({a_name}@{a_pos}) hp={a_hp} -> def_id={d_uid} ({d_name}@{d_pos}) hp={d_hp}")
        for k, v in per_pid_ci.items():
            print(f"        per-pid {k}: att_hp={v['att_hp']} def_hp={v['def_hp']}")

    # Also confirm the multi-hit defenders in env 17
    def_hits = {}
    for obj in actions:
        if obj.get("action") not in ("Fire", "AttackSeam"):
            continue
        ci = obj.get("combatInfo") or {}
        d = ci.get("defender") if isinstance(ci, dict) else None
        if isinstance(d, dict):
            try:
                d_uid = int(d["units_id"])
                def_hits[d_uid] = def_hits.get(d_uid, 0) + 1
            except Exception:
                pass
    multi = {uid for uid, c in def_hits.items() if c > 1}
    print(f"\nmulti-hit defenders in env 17: {multi}")
    print(f"all def_hits in env 17: {def_hits}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
