"""Compare PHP frame 17 (pre-env-17) vs engine state mid-env-17 for specific units."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    parse_p_envelopes_from_zip,
    apply_oracle_action_json,
    resolve_replay_first_mover,
    map_snapshot_player_ids_to_engine,
)
from engine.map_loader import load_map
from engine.game import make_initial_state


# Map engine id -> PHP id for affected units (from prior dump)
TARGETS = {2: 190277871, 6: 190289865}


def main() -> int:
    gid = 1607045
    co0, co1 = 5, 28
    map_id = 77060
    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    map_pool = REPO / "data/gl_map_pool.json"
    maps_dir = REPO / "data/maps"

    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    map_data = load_map(map_id, map_pool, maps_dir)
    state = make_initial_state(map_data, co0, co1, starting_funds=0, tier_name="T1", replay_first_mover=first_mover)

    def setup_pin(env_i, actions):
        if env_i + 1 >= len(frames):
            state._oracle_post_envelope_units_by_id = None
            state._oracle_post_envelope_multi_hit_defenders = None
            return
        pin = {}
        for u in (frames[env_i + 1].get("units") or {}).values():
            try:
                uid = int(u["id"]); hp = float(u["hit_points"])
                pin[uid] = max(0, min(100, int(round(hp * 10))))
            except Exception:
                pass
        def_hits = {}
        for obj in actions:
            if isinstance(obj, dict) and obj.get("action") in ("Fire", "AttackSeam"):
                ci = obj.get("combatInfo") or {}
                d = ci.get("defender") if isinstance(ci, dict) else None
                if isinstance(d, dict):
                    try:
                        def_hits[int(d["units_id"])] = def_hits.get(int(d["units_id"]), 0) + 1
                    except Exception:
                        pass
        state._oracle_post_envelope_units_by_id = pin
        state._oracle_post_envelope_multi_hit_defenders = {uid for uid, c in def_hits.items() if c > 1}

    def php_unit(frame_idx, php_id):
        for u in (frames[frame_idx].get("units") or {}).values():
            try:
                if int(u["id"]) == php_id:
                    return u
            except Exception:
                continue
        return None

    def engine_unit(eng_id):
        for p in (0, 1):
            for u in state.units[p]:
                if u.unit_id == eng_id and u.is_alive:
                    return u
        return None

    for i, (pid, day, actions) in enumerate(envs):
        if i == 17:
            print("=== State BEFORE env 17 ===")
            for eid, php_id in TARGETS.items():
                e = engine_unit(eid)
                p_pre = php_unit(17, php_id)  # frame 17 = state at start of env 17
                p_post = php_unit(18, php_id)  # frame 18 = state at start of env 18
                e_hp = e.hp if e else None
                p_pre_hp = float(p_pre["hit_points"]) if p_pre else None
                p_post_hp = float(p_post["hit_points"]) if p_post else None
                print(f"  eng_id={eid} php_id={php_id}: engine_hp={e_hp}  PHP_pre_env_17={p_pre_hp}  PHP_post_env_17={p_post_hp}")

            # Apply env 17 with pin
            setup_pin(i, actions)
            for obj in actions:
                if state.done:
                    break
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)

            print("=== State AFTER env 17 ===")
            for eid, php_id in TARGETS.items():
                e = engine_unit(eid)
                p_post = php_unit(18, php_id)
                e_hp = e.hp if e else None
                p_post_hp = float(p_post["hit_points"]) if p_post else None
                print(f"  eng_id={eid} php_id={php_id}: engine_hp={e_hp}  PHP_post={p_post_hp}")
            return 0
        else:
            setup_pin(i, actions)
            for obj in actions:
                if state.done:
                    break
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)

    return 0


if __name__ == "__main__":
    sys.exit(main())
