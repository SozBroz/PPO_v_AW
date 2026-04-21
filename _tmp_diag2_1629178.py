"""Apply everything up to envelope 30 action 9 (AttackSeam) and inspect engine state."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.action import Action, ActionType, get_legal_actions, get_attack_targets, compute_reachable_costs  # noqa: E402
from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
    UnsupportedOracleAction,
)
from tools.diff_replay_zips import load_replay  # noqa: E402

GID = 1629178
TARGET_ENV = 30
TARGET_ACT = 9  # AttackSeam
ZP = ROOT / "replays" / "amarriner_gl" / f"{GID}.zip"

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _meta_for(gid: int) -> dict:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    for _k, g in (cat.get("games") or {}).items():
        if isinstance(g, dict) and int(g.get("games_id", -1)) == gid:
            return g
    raise KeyError(gid)


def main() -> int:
    meta = _meta_for(GID)
    map_id = int(meta["map_id"])
    co0 = int(meta["co_p0_id"])
    co1 = int(meta["co_p1_id"])
    tier = str(meta.get("tier") or "global_league")

    map_data = load_map(map_id, MAP_POOL, MAPS_DIR)
    frames = load_replay(ZP)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    envs = parse_p_envelopes_from_zip(ZP)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data, co0, co1, starting_funds=0, tier_name=tier,
        replay_first_mover=first_mover,
    )

    # Apply envelopes 0..TARGET_ENV-1
    for env_i, (pid, day, actions) in enumerate(envs):
        if env_i >= TARGET_ENV:
            break
        for obj in actions:
            if state.done:
                break
            apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)

    # Apply TARGET_ENV actions 0..TARGET_ACT-1
    pid, day, actions = envs[TARGET_ENV]
    for ai in range(TARGET_ACT):
        try:
            apply_oracle_action_json(state, actions[ai], awbw_to_engine, envelope_awbw_player_id=pid)
            print(f"OK env_i={TARGET_ENV} act={ai} kind={actions[ai].get('action')}")
        except UnsupportedOracleAction as e:
            print(f"FAIL env_i={TARGET_ENV} act={ai} kind={actions[ai].get('action')}: {e}")
            return 2

    print("\n=== state at env_i=30 act=9 (AttackSeam) ===")
    print(f"active_player={state.active_player}")
    print("seam_hp:", dict(state.seam_hp))

    # Look at the seam (7,6) and infantry at (4,6)
    sr, sc = 7, 6
    print(f"\nterrain near seam ({sr},{sc}):")
    for r in range(max(0, sr-3), min(state.map_data.height, sr+4)):
        row = []
        for c in range(max(0, sc-3), min(state.map_data.width, sc+4)):
            tid = state.map_data.terrain[r][c]
            row.append(f"{tid:3d}")
        print(f"  r={r}: {' '.join(row)}")

    print(f"\nunits in window:")
    for p in (0, 1):
        for u in state.units[p]:
            r, c = u.pos
            if max(0, sr-3) <= r <= min(state.map_data.height-1, sr+3) and max(0, sc-3) <= c <= min(state.map_data.width-1, sc+3):
                print(f"  P{p} uid={u.unit_id} type={u.unit_type.name} pos={u.pos} hp={u.hp} ammo={u.ammo} moved={u.moved} alive={u.is_alive}")

    # Now perform the SELECT and look at legal actions
    obj = actions[TARGET_ACT]
    print(f"\nAttackSeam JSON:")
    print(f"  seam_xy: ({obj['AttackSeam']['seamY']}, {obj['AttackSeam']['seamX']})")
    paths = (obj.get("Move", {}).get("paths") or {}).get("global") or []
    print(f"  paths.global: {paths}")
    if paths:
        sr_path, sc_path = int(paths[0]["y"]), int(paths[0]["x"])
        er_path, ec_path = int(paths[-1]["y"]), int(paths[-1]["x"])
        u_at_start = state.get_unit_at(sr_path, sc_path)
        if u_at_start:
            from engine.combat import get_seam_base_damage
            print(f"\nunit at path start ({sr_path},{sc_path}): uid={u_at_start.unit_id} type={u_at_start.unit_type.name} player={u_at_start.player} hp={u_at_start.hp} ammo={u_at_start.ammo} moved={u_at_start.moved}")
            print(f"seam_base_damage for {u_at_start.unit_type.name}: {get_seam_base_damage(u_at_start.unit_type)}")
            reach = compute_reachable_costs(state, u_at_start)
            print(f"reach (cost) keys: {sorted(reach.keys())}")
            for wp in paths:
                pos = (int(wp["y"]), int(wp["x"]))
                in_reach = pos in reach
                tgts = get_attack_targets(state, u_at_start, pos) if in_reach else []
                seam_in = (sr, sc) in tgts
                print(f"  waypoint {pos}: reach={in_reach} attack_targets_count={len(tgts)} seam_in_targets={seam_in}")
                if tgts:
                    for t in tgts:
                        tid = state.map_data.terrain[t[0]][t[1]]
                        u_t = state.get_unit_at(*t)
                        print(f"    target {t} terrain_id={tid} unit_at={u_t.unit_type.name if u_t else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
