"""Strict Gold Rush ×1.5 drill against AWBW canonical wiki/co.php rule.

For every Colin COP envelope with sub_index == 0 across the scraped zips:
  pre  = PHP frame[k] funds for Colin's awbw players_id (state BEFORE envelope)
  post_action  = Power.playerReplace.global[<id>].players_funds  (server payload)
  post_frame   = PHP frame[k+1] funds for the same player        (state AFTER envelope)
  expected     = int(pre * 1.5)

We require post_action == expected. We additionally surface post_frame == expected
when the envelope contains no funds-affecting action besides Power+End (Build,
Repair, etc. would change post_frame).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.oracle_zip_replay import parse_p_envelopes_from_zip  # noqa: E402

GIDS = [
    1636107, 1558571, 1637153, 1638360, 1636411, 1620117, 1628024,
    1629555, 1358720, 1636108, 1619141, 1637200,
]

FUNDS_TOUCHING_ACTIONS = {
    "Build", "Repair", "Power", "Move", "Capt", "Fire", "AttackSeam", "Load",
    "Unload", "Supply", "Hide", "Unhide", "Delete", "Join", "End", "GameOver",
}
# Conservative: Build / Repair definitely change funds; treat everything else as
# possibly funds-neutral and report observed post_frame value.
FUNDS_DEFINITELY_TOUCHED = {"Build", "Repair"}


def colin_pids(frame0: dict) -> set[int]:
    out: set[int] = set()
    for p in (frame0.get("players") or {}).values():
        if isinstance(p, dict) and int(p.get("co_id", 0)) == 15:
            out.add(int(p["id"]))
    return out


def round_half_up(x: float) -> int:
    """AWBW uses round-half-up for ×1.5 funds (PHP `round()` default)."""
    import math
    return int(math.floor(x + 0.5))


def funds_for(frame: dict, pid: int) -> int:
    for p in (frame.get("players") or {}).values():
        if isinstance(p, dict) and int(p.get("id", -1)) == pid:
            return int(p.get("funds", 0) or 0)
    return -1


def drill_zip(zpath: Path) -> list[dict]:
    frames = load_replay(zpath)
    envs = parse_p_envelopes_from_zip(zpath)
    pids = colin_pids(frames[0]) if frames else set()
    if not pids or not envs:
        return []
    out: list[dict] = []
    for k, (_p, _d, actions) in enumerate(envs):
        if k + 1 >= len(frames):
            continue
        for sub_idx, a in enumerate(actions):
            if a.get("action") != "Power" or a.get("coName") != "Colin":
                continue
            if a.get("coPower") != "Y":
                continue
            if sub_idx != 0:
                continue
            # Dispatch to the SPECIFIC Colin who fired the COP (Power.playerID),
            # else fall back to the envelope owner pid. This matters in Colin-vs-Colin.
            actor = a.get("playerID")
            if actor is None:
                actor = _p
            actor = int(actor)
            if actor not in pids:
                # actor pid not present in opening frame players block; skip
                continue
            pre = funds_for(frames[k], actor)
            post_action = (
                ((a.get("playerReplace") or {}).get("global") or {})
                .get(str(actor), {})
                .get("players_funds")
            )
            post_frame = funds_for(frames[k + 1], actor)
            expected = round_half_up(pre * 1.5)
            expected_floor = int(pre * 1.5)
            kinds = [str(x.get("action", "")) for x in actions]
            funds_polluted = any(k_ in FUNDS_DEFINITELY_TOUCHED for k_ in kinds[1:])
            out.append(
                {
                    "zip": zpath.name,
                    "env_k": k,
                    "actor_pid": actor,
                    "actions": kinds,
                    "funds_polluted_after_power": funds_polluted,
                    "pre": pre,
                    "expected_round_half_up": expected,
                    "expected_floor": expected_floor,
                    "post_action_payload": post_action,
                    "post_frame": post_frame,
                    "match_round_half_up": post_action == expected,
                    "match_floor": post_action == expected_floor,
                    "match_frame": (post_frame == expected) if not funds_polluted else None,
                }
            )
    return out


def main() -> None:
    base = ROOT / "replays" / "amarriner_gl"
    rows: list[dict] = []
    for gid in GIDS:
        zpath = base / f"{gid}.zip"
        if not zpath.is_file():
            continue
        rows.extend(drill_zip(zpath))

    n = len(rows)
    rows_with_payload = [r for r in rows if r["post_action_payload"] is not None]
    pm_round = sum(1 for r in rows_with_payload if r["match_round_half_up"])
    pm_floor = sum(1 for r in rows_with_payload if r["match_floor"])
    print(f"sub=0 Colin COP envelopes scanned: {n}")
    print(f"  envelopes carrying playerReplace.players_funds: {len(rows_with_payload)}")
    print(f"    payload == round_half_up(pre*1.5): {pm_round}/{len(rows_with_payload)}")
    print(f"    payload == int(pre*1.5)         : {pm_floor}/{len(rows_with_payload)}")
    print()
    for r in rows:
        if r["post_action_payload"] is None:
            tag = "[no playerReplace.players_funds in COP JSON]"
        elif r["match_round_half_up"]:
            tag = "[OK round_half_up]"
        elif r["match_floor"]:
            tag = "[OK floor]"
        else:
            tag = "[MISMATCH]"
        print(
            f"{r['zip']} env={r['env_k']:3d} pid={r['actor_pid']} pre={r['pre']:6d} "
            f"exp_round={r['expected_round_half_up']:6d} exp_floor={r['expected_floor']:6d} "
            f"payload={r['post_action_payload']!s:>7s} {tag} "
            f"frame[k+1]={r['post_frame']:6d}"
        )


if __name__ == "__main__":
    main()
