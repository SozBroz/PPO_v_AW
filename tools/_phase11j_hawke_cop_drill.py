"""Drill Hawke COP/SCOP HP-delta truth from PHP snapshots.

Phase 11J-FINAL-HAWKE-CLUSTER-OWNER. Compares PHP frame[N] vs frame[N+1]
HP for each unit around a Hawke power envelope to verify the canon
+1 HP COP / +2 HP SCOP friend heal and -1 / -2 enemy damage.

Usage:
  python tools/_phase11j_hawke_cop_drill.py [<gid> [<env_index>]]

Defaults: 1635846 env 30 (Hawke COP "Black Wave").
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay  # type: ignore
from tools.oracle_zip_replay import parse_p_envelopes_from_zip  # type: ignore


def _units_from_frame(frame: dict) -> dict[int, dict]:
    units = frame.get("units", {})
    out: dict[int, dict] = {}
    for k, u in units.items():
        if isinstance(u, dict):
            try:
                uid = int(u.get("id"))
            except (TypeError, ValueError):
                continue
            out[uid] = u
    return out


def _hp_pairs(prev_units: dict[int, dict], next_units: dict[int, dict]) -> list[tuple[int, dict, dict]]:
    out = []
    for uid, prev in prev_units.items():
        nxt = next_units.get(uid)
        if nxt is None:
            continue
        out.append((uid, prev, nxt))
    return out


def main() -> int:
    gid = int(sys.argv[1]) if len(sys.argv) > 1 else 1635846
    target_env = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    path = REPO / "replays" / "amarriner_gl" / f"{gid}.zip"
    if not path.exists():
        print(f"MISSING {path}")
        return 1
    frames = load_replay(path)
    envelopes = parse_p_envelopes_from_zip(path)
    print(f"gid={gid} frames={len(frames)} envelopes={len(envelopes)} target_env={target_env}")
    if target_env >= len(envelopes):
        print(f"  envelope index out of range")
        return 1
    pid, day, actions = envelopes[target_env]
    power = next((a for a in actions if a.get("action") == "Power"), None)
    if power is None:
        print(f"  envelope {target_env} has no Power action")
        return 1
    print(f"  power: pid={pid} day={day} co={power.get('coName')!r} "
          f"name={power.get('powerName')!r} flag={power.get('coPower')!r}")
    # PHP frames are aligned with envelopes (one frame per turn-snapshot).
    # The frame index that holds the snapshot AFTER envelope N is typically
    # related to the day. Try both N and N+1 to inspect.
    for label, idx in (("frame[env]", target_env), ("frame[env+1]", target_env + 1)):
        if idx < 0 or idx >= len(frames):
            continue
        units = _units_from_frame(frames[idx])
        own = sum(1 for u in units.values() if u.get("players_id") == pid)
        enemy = sum(1 for u in units.values() if u.get("players_id") != pid)
        print(f"  {label} idx={idx} day={frames[idx].get('day')} active={frames[idx].get('turn')} "
              f"units(own={own}, enemy={enemy})")
    # Compare the two frames: envelopes are 0-indexed but frames may use day index
    # The simple model: snapshot N corresponds to STATE BEFORE envelope N.
    if target_env + 1 >= len(frames):
        print("  cannot diff (no frame[env+1])")
        return 0
    prev_units = _units_from_frame(frames[target_env])
    next_units = _units_from_frame(frames[target_env + 1])
    print("\n  unit HP deltas across envelope (units present in both frames):")
    own_hp_deltas: list[float] = []
    enemy_hp_deltas: list[float] = []
    for uid, prev, nxt in _hp_pairs(prev_units, next_units):
        prev_hp = prev.get("hit_points")
        next_hp = nxt.get("hit_points")
        if prev_hp is None or next_hp is None:
            continue
        try:
            prev_d = float(prev_hp)
            next_d = float(next_hp)
        except (TypeError, ValueError):
            continue
        delta = round(next_d - prev_d, 2)
        u_pid = prev.get("players_id")
        if abs(delta) < 0.05:
            continue
        side = "OWN" if u_pid == pid else "ENEMY"
        name = prev.get("name")
        pos = (prev.get("y"), prev.get("x"))
        print(f"    {side} uid={uid:>9} {name:<10} pos={pos} hp {prev_d:5.1f} -> {next_d:5.1f} (Δ {delta:+5.1f})")
        if u_pid == pid:
            own_hp_deltas.append(delta)
        else:
            enemy_hp_deltas.append(delta)
    print(f"\n  summary: own n={len(own_hp_deltas)} mean={sum(own_hp_deltas)/max(1,len(own_hp_deltas)):+.2f}  "
          f"enemy n={len(enemy_hp_deltas)} mean={sum(enemy_hp_deltas)/max(1,len(enemy_hp_deltas)):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
