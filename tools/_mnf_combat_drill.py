"""Drill into a single Fire action: dump engine pre-state, terrain, CO, formula step.

Usage: python tools/_mnf_combat_drill.py 1632825 <env_idx> <ai_idx>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from engine.combat import calculate_damage, get_base_damage
from engine.terrain import get_terrain
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


def _find_unit_at(state, x, y):
    for seat, lst in state.units.items():
        for u in lst:
            if u.is_alive and tuple(u.pos) == (y, x):
                return seat, u
    return None, None


def main() -> None:
    gid = int(sys.argv[1])
    target_env = int(sys.argv[2])
    target_ai = int(sys.argv[3])
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

    for env_idx, (pid, day, actions) in enumerate(envs):
        for ai, obj in enumerate(actions):
            if state.done:
                break
            if env_idx == target_env and ai == target_ai:
                kind = obj.get("action")
                print(f"=== TARGET env={env_idx} ai={ai} pid={pid} day={day} kind={kind} ===")
                print(f"state.active_player={state.active_player}")
                print(f"co_p0_id={state.co_states[0].co_id} cop_active={state.co_states[0].cop_active} scop_active={state.co_states[0].scop_active} power_bar={state.co_states[0].power_bar}")
                print(f"co_p1_id={state.co_states[1].co_id} cop_active={state.co_states[1].cop_active} scop_active={state.co_states[1].scop_active} power_bar={state.co_states[1].power_bar}")
                fire = obj.get("Fire") or {}
                # apply embedded Move first if any
                emb_move = obj.get("Move") or []
                # Use civ to find tiles
                civ = fire.get("combatInfoVision") or {}
                ax = ay = dx = dy = None
                for view in civ.values():
                    ci = view.get("combatInfo") if isinstance(view, dict) else None
                    if not isinstance(ci, dict):
                        continue
                    a = ci.get("attacker") or {}
                    d = ci.get("defender") or {}
                    if ax is None and a.get("units_x") is not None:
                        ax, ay = a.get("units_x"), a.get("units_y")
                    if dx is None and d.get("units_x") is not None:
                        dx, dy = d.get("units_x"), d.get("units_y")
                print(f"ATK tile=({ax},{ay}) DEF tile=({dx},{dy})")
                # find units at those tiles in current state
                _, atk = _find_unit_at(state, ax, ay)
                _, deff = _find_unit_at(state, dx, dy)
                print(f"ATK in engine: {atk}")
                print(f"DEF in engine: {deff}")
                # Try also pre-Move attacker location: search by AWBW uid
                aw_atk_uid = aw_def_uid = None
                for view in civ.values():
                    ci = view.get("combatInfo") if isinstance(view, dict) else None
                    if not isinstance(ci, dict):
                        continue
                    a = ci.get("attacker") or {}
                    d = ci.get("defender") or {}
                    if aw_atk_uid is None:
                        aw_atk_uid = a.get("units_id")
                    if aw_def_uid is None:
                        aw_def_uid = d.get("units_id")
                print(f"AWBW atk_uid={aw_atk_uid} def_uid={aw_def_uid}")
                # find by AWBW uid (if oracle stored mapping)
                # Inspect terrain
                if dx is not None:
                    terr = get_terrain(map_data.terrain[dy][dx])
                    print(f"Defender terrain ({dx},{dy}) row={dy} col={dx}: {terr.name} stars={terr.defense}")
                if ax is not None:
                    terr = get_terrain(map_data.terrain[ay][ax])
                    print(f"Attacker terrain ({ax},{ay}) row={ay} col={ax}: {terr.name} stars={terr.defense}")
                # Hand-calc damage with luck=0
                if atk and deff:
                    base = get_base_damage(atk.unit_type, deff.unit_type)
                    print(f"base[{atk.unit_type.name}->{deff.unit_type.name}] = {base}")
                    for roll in range(10):
                        dmg = calculate_damage(
                            atk, deff,
                            get_terrain(map_data.terrain[atk.pos[0]][atk.pos[1]]),
                            get_terrain(map_data.terrain[deff.pos[0]][deff.pos[1]]),
                            state.co_states[atk.player], state.co_states[deff.player],
                            luck_roll=roll,
                        )
                        print(f"  engine luck={roll} dmg={dmg} -> def_post_int={max(0, deff.hp - dmg)} disp={(max(0, deff.hp - dmg)+9)//10}")
                # Print AWBW reported HPs (from civ) for cross-check
                print("=== AWBW combatInfoVision ===")
                print(json.dumps(civ, indent=2, default=str)[:3000])
                return
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    before_engine_step=None,
                    envelope_awbw_player_id=pid,
                )
            except UnsupportedOracleAction as e:
                print(f"FAIL before target env={env_idx} ai={ai}: {e}")
                return


if __name__ == "__main__":
    main()
