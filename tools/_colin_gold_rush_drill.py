"""Gold Rush ×1.5 funds check: PHP frame[k] vs frame[k+1] at COP envelope index."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.oracle_zip_replay import parse_p_envelopes_from_zip  # noqa: E402


def colin_players_id(frames: list[dict], *, want_co_id: int = 15) -> int | None:
    f0 = frames[0]
    players = f0.get("players") or {}
    for _k, pl in players.items():
        if not isinstance(pl, dict):
            continue
        if int(pl.get("co_id", 0)) == want_co_id:
            return int(pl["id"])
    return None


def funds_for_pid(frame: dict, pid: int) -> int:
    players = frame.get("players") or {}
    for _k, pl in players.items():
        if not isinstance(pl, dict):
            continue
        if int(pl.get("id", -1)) == pid:
            return int(pl.get("funds", 0) or 0)
    return -1


def first_cop_env_index(zpath: Path) -> int | None:
    envs = parse_p_envelopes_from_zip(zpath)
    for ei, (_pid, _day, actions) in enumerate(envs):
        for a in actions:
            if (
                a.get("action") == "Power"
                and a.get("coName") == "Colin"
                and a.get("coPower") == "Y"
            ):
                return ei
    return None


def drill(games_id: int) -> None:
    zpath = ROOT / "replays" / "amarriner_gl" / f"{games_id}.zip"
    frames = load_replay(zpath)
    envs = parse_p_envelopes_from_zip(zpath)
    print(f"games_id={games_id} frames={len(frames)} envelopes={len(envs)}")
    pid = colin_players_id(frames)
    print(f"  Colin awbw players_id={pid}")
    if pid is None or len(frames) < 2:
        print("  abort")
        return
    k = first_cop_env_index(zpath)
    if k is None:
        print("  no Colin COP envelope")
        return
    print(f"  first Colin COP at envelope index k={k}")
    if k + 1 >= len(frames):
        print("  no frame[k+1]")
        return
    pre = funds_for_pid(frames[k], pid)
    post = funds_for_pid(frames[k + 1], pid)
    exp = int(pre * 1.5)
    ratio = post / pre if pre else None
    print(f"  funds frame[k]={pre} frame[k+1]={post} expected int(pre*1.5)={exp}")
    print(f"  match={post == exp} ratio_post/pre={ratio}")


def main() -> None:
    for gid in (1636107, 1636108):
        drill(gid)
        print()


if __name__ == "__main__":
    main()
