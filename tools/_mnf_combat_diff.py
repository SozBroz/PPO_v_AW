"""Per-Fire diff: AWBW reported post-fight HP vs engine pre/post HP for a games_id.

Walks the oracle replay envelope-by-envelope. At every Fire action, before
applying it to the engine state, captures the engine attacker/defender HP
(matching by tile coords from the AWBW Fire combatInfo). Then applies the
action and captures post-fight HP. Compares engine post HP vs AWBW post HP
(display HP from defender-side combatInfoVision when global is fog ?, else
from global). Also reports CO power state at the time of fire.

Usage:  python tools/_mnf_combat_diff.py 1632825
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    apply_oracle_action_json,
    load_map,
    load_replay,
    make_initial_state,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)


def _catalog_lookup(gid: int):
    for cat in (
        Path("data/amarriner_gl_std_catalog.json"),
        Path("data/amarriner_gl_extras_catalog.json"),
    ):
        if not cat.exists():
            continue
        d = json.loads(cat.read_text(encoding="utf-8"))
        games = d.get("games", d) if isinstance(d, dict) else d
        if isinstance(games, dict):
            games = list(games.values())
        for r in games:
            if int(r.get("games_id", 0)) == gid:
                return r
    raise SystemExit("missing catalog row")


def _aw_post_hp(combat_info: dict[str, Any]) -> tuple[int | None, int | None]:
    """Pull (attacker_hp, defender_hp) post-fight from AWBW combatInfoVision.

    Prefer the seat-resolved view that has numeric (no '?'): try every per-pid
    block (and 'global'); collect the highest-information numeric hp for each
    side. (Owners always see numeric HP for their own units.)
    """
    atk_hp: int | None = None
    def_hp: int | None = None
    for key, view in combat_info.items():
        ci = view.get("combatInfo") if isinstance(view, dict) else None
        if not isinstance(ci, dict):
            continue
        a = ci.get("attacker") or {}
        d = ci.get("defender") or {}
        a_hp = a.get("units_hit_points")
        d_hp = d.get("units_hit_points")
        if isinstance(a_hp, int) and atk_hp is None:
            atk_hp = a_hp
        if isinstance(d_hp, int) and def_hp is None:
            def_hp = d_hp
    return atk_hp, def_hp


def _find_unit_at(state, x: int, y: int):
    for seat, lst in state.units.items():
        for u in lst:
            if u.is_alive and tuple(u.pos) == (y, x):  # engine pos = (row, col)
                return seat, u
    return None, None


def main() -> None:
    gid = int(sys.argv[1])
    target_uid = int(sys.argv[2]) if len(sys.argv) > 2 else None
    row = _catalog_lookup(gid)
    zip_path = Path(f"replays/amarriner_gl/{gid}.zip")
    frames = load_replay(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(
        frames[0], int(row["co_p0_id"]), int(row["co_p1_id"]),
    )
    map_data = load_map(int(row["map_id"]), Path("data/gl_map_pool.json"), Path("data/maps"))
    envs = parse_p_envelopes_from_zip(zip_path)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data, int(row["co_p0_id"]), int(row["co_p1_id"]),
        starting_funds=0, tier_name=str(row["tier"]),
        replay_first_mover=first_mover,
    )

    n_act = 0
    for env_idx, (pid, day, actions) in enumerate(envs):
        for ai, obj in enumerate(actions):
            if state.done:
                break
            kind = obj.get("action")
            # Capture engine snapshot BEFORE applying the Fire action.
            pre_snapshot: dict[str, Any] | None = None
            if kind == "Fire":
                fire = obj.get("Fire") or {}
                civ = fire.get("combatInfoVision") or {}
                # Resolve attacker/defender tiles from any view that has them.
                ax = ay = dx = dy = None
                aw_uid_a = aw_uid_d = None
                for view in civ.values():
                    ci = view.get("combatInfo") if isinstance(view, dict) else None
                    if not isinstance(ci, dict):
                        continue
                    a = ci.get("attacker") or {}
                    d = ci.get("defender") or {}
                    if ax is None and a.get("units_x") is not None:
                        ax, ay = a.get("units_x"), a.get("units_y")
                        aw_uid_a = a.get("units_id")
                    if dx is None and d.get("units_x") is not None:
                        dx, dy = d.get("units_x"), d.get("units_y")
                        aw_uid_d = d.get("units_id")
                eng_atk = eng_def = None
                if ax is not None:
                    _, eng_atk = _find_unit_at(state, ax, ay)
                if dx is not None:
                    _, eng_def = _find_unit_at(state, dx, dy)
                pre_snapshot = {
                    "atk": (ax, ay, aw_uid_a, eng_atk.hp if eng_atk else None,
                            eng_atk.unit_type.name if eng_atk else None,
                            int(eng_atk.player) if eng_atk else None),
                    "def": (dx, dy, aw_uid_d, eng_def.hp if eng_def else None,
                            eng_def.unit_type.name if eng_def else None,
                            int(eng_def.player) if eng_def else None),
                }
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    before_engine_step=None,
                    envelope_awbw_player_id=pid,
                )
                n_act += 1
            except UnsupportedOracleAction as e:
                print(f"\nFAIL env={env_idx} ai={ai} day={day}: {e}")
                return
            if kind == "Fire" and pre_snapshot is not None:
                fire = obj.get("Fire") or {}
                civ = fire.get("combatInfoVision") or {}
                aw_atk_post, aw_def_post = _aw_post_hp(civ)
                # Engine post: re-find by AWBW uid first (preferred), else by tile.
                eng_atk_post = eng_def_post = None
                ax, ay, aw_uid_a, *_ = pre_snapshot["atk"]
                dx, dy, aw_uid_d, *_ = pre_snapshot["def"]
                _, eng_atk = _find_unit_at(state, ax, ay)
                if eng_atk:
                    eng_atk_post = eng_atk.hp if eng_atk.is_alive else 0
                _, eng_def = _find_unit_at(state, dx, dy)
                if eng_def:
                    eng_def_post = eng_def.hp if eng_def.is_alive else 0
                # Convert to display HP for direct compare with AWBW
                def disp(h):
                    if h is None:
                        return None
                    if h <= 0:
                        return 0
                    return (h + 9) // 10
                e_a = disp(eng_atk_post)
                e_d = disp(eng_def_post)
                marker = ""
                if aw_atk_post is not None and e_a is not None and aw_atk_post != e_a:
                    marker += " ATK_DIFF"
                if aw_def_post is not None and e_d is not None and aw_def_post != e_d:
                    marker += " DEF_DIFF"
                if (target_uid is None) or (
                    aw_uid_a == target_uid or aw_uid_d == target_uid or marker
                ):
                    pre_atk = pre_snapshot["atk"]
                    pre_def = pre_snapshot["def"]
                    print(
                        f"env={env_idx:>2} ai={ai:>2} day={day} pid={pid} "
                        f"ATK uid={aw_uid_a} {pre_atk[4]} P{pre_atk[5]} ({ax},{ay}) "
                        f"hp_pre={pre_atk[3]} aw_post_disp={aw_atk_post} eng_post_disp={e_a} (eng_post_int={eng_atk_post}) | "
                        f"DEF uid={aw_uid_d} {pre_def[4]} P{pre_def[5]} ({dx},{dy}) "
                        f"hp_pre={pre_def[3]} aw_post_disp={aw_def_post} eng_post_disp={e_d} (eng_post_int={eng_def_post})"
                        f"{marker}"
                    )


if __name__ == "__main__":
    main()
