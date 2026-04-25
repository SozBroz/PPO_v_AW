"""Compare engine (with pin) vs PHP units around env 17 of gid 1607045."""
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


def setup_pin(state, frames, env_i, actions):
    if env_i + 1 >= len(frames):
        state._oracle_post_envelope_units_by_id = None
        state._oracle_post_envelope_multi_hit_defenders = None
        return
    pin = {}
    for u in (frames[env_i + 1].get("units") or {}).values():
        try:
            uid = int(u["id"])
            hp = float(u["hit_points"])
            pin[uid] = max(0, min(100, int(round(hp * 10))))
        except Exception:
            pass
    end_rep = set()
    for obj in actions:
        if isinstance(obj, dict) and obj.get("action") == "End":
            ui = obj.get("updatedInfo") or {}
            rep = ui.get("repaired") if isinstance(ui, dict) else None
            if isinstance(rep, dict): rep = rep.get("global")
            if isinstance(rep, list):
                for r in rep:
                    if isinstance(r, dict):
                        try:
                            end_rep.add(int(r.get("units_id")))
                        except Exception:
                            pass
    for uid in end_rep:
        pin.pop(uid, None)
    def_hits = {}
    for obj in actions:
        if isinstance(obj, dict) and obj.get("action") in ("Fire", "AttackSeam"):
            ci = obj.get("combatInfo")
            if isinstance(ci, dict):
                d = ci.get("defender")
                if isinstance(d, dict):
                    try:
                        d_uid = int(d.get("units_id"))
                        def_hits[d_uid] = def_hits.get(d_uid, 0) + 1
                    except Exception:
                        pass
    state._oracle_post_envelope_units_by_id = pin
    state._oracle_post_envelope_multi_hit_defenders = {uid for uid, c in def_hits.items() if c > 1}


def dump_engine_units(state, label):
    print(f"\n--- ENGINE units {label} ---")
    print(f"  funds P0={state.funds[0]} P1={state.funds[1]} active={state.active_player}")
    for p in (0, 1):
        for u in state.units[p]:
            if not u.is_alive:
                continue
            print(f"  P{p} id={u.unit_id:>10} {u.unit_type.name:<10} pos={u.pos} hp={u.hp:>3}")


def dump_php_units(frame, awbw_to_engine, label):
    print(f"\n--- PHP units (frame {label}) ---")
    print(f"  day={frame.get('day')} active_player(awbw)={frame.get('activePlayerId')}")
    by_seat = {0: [], 1: []}
    for u in (frame.get("units") or {}).values():
        try:
            pid = int(u.get("players_id") or u.get("units_players_id") or u.get("player_id"))
        except Exception:
            continue
        eng = awbw_to_engine.get(pid)
        if eng is None:
            continue
        try:
            uid = int(u["id"])
            hp = float(u.get("hit_points"))
            x = int(u.get("x") or u.get("units_x") or 0)
            y = int(u.get("y") or u.get("units_y") or 0)
            name = u.get("name") or u.get("units_name") or "?"
        except Exception:
            continue
        by_seat[eng].append((uid, name, (y, x), hp))
    for s in (0, 1):
        for uid, name, pos, hp in by_seat[s]:
            internal = int(round(hp * 10))
            print(f"  P{s} id={uid:>10} {name:<10} pos={pos} hit_points={hp:.2f} internal={internal}")


def main() -> int:
    gid = 1607045
    co0, co1 = 5, 28
    map_id = 77060
    target_env = int(sys.argv[1]) if len(sys.argv) > 1 else 17

    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    map_pool = REPO / "data/gl_map_pool.json"
    maps_dir = REPO / "data/maps"

    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    map_data = load_map(map_id, map_pool, maps_dir)
    state = make_initial_state(map_data, co0, co1, starting_funds=0, tier_name="T1", replay_first_mover=first_mover)

    for i, (pid, day, actions) in enumerate(envs):
        setup_pin(state, frames, i, actions)
        for obj in actions:
            if state.done:
                break
            apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
        if i == target_env:
            dump_engine_units(state, f"after env {i}")
            dump_php_units(frames[i + 1], awbw_to_engine, f"{i+1}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
