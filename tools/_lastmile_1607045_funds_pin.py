"""Funds trace for gid 1607045 WITH the post-envelope HP pin (audit path)."""
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
    state = make_initial_state(map_data, co0, co1, starting_funds=0,
                               tier_name="T1", replay_first_mover=first_mover)

    def php_funds(frame: dict) -> tuple[int, int]:
        out = [0, 0]
        for k, p in (frame.get("players") or {}).items() if isinstance(frame.get("players"), dict) else []:
            try:
                pid = int(p.get("id"))
                eng = awbw_to_engine.get(pid)
                if eng is not None:
                    out[eng] = int(p.get("funds", 0))
            except Exception:
                pass
        return tuple(out)

    print(f"{'env':>3} {'day':>3} {'pid':>9} {'eng_p0':>7} {'eng_p1':>7} | {'php_p0':>7} {'php_p1':>7} | {'d_p0':>5} {'d_p1':>5} pin_units multi_def kinds")

    for i, (pid, day, actions) in enumerate(envs):
        post_frame = frames[i + 1] if (i + 1) < len(frames) else None
        pin = {}
        if post_frame is not None:
            for u in (post_frame.get("units") or {}).values():
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
                if isinstance(rep, dict):
                    rep = rep.get("global")
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
        multi = {uid for uid, c in def_hits.items() if c > 1}
        state._oracle_post_envelope_units_by_id = pin
        state._oracle_post_envelope_multi_hit_defenders = multi

        for obj in actions:
            if state.done:
                break
            try:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
            except Exception as e:
                print(f"  env={i} ABORT {type(e).__name__}: {e}")
                return 1
        eng = (state.funds[0], state.funds[1])
        php_idx = i + 1 if (i + 1) < len(frames) else i
        ph = php_funds(frames[php_idx])
        d0, d1 = eng[0] - ph[0], eng[1] - ph[1]
        flag = "!!!!" if (abs(d0) >= 50 or abs(d1) >= 50) else "    "
        kinds = ",".join(sorted({(o.get("action") or "?") for o in actions if isinstance(o, dict)}))[:48]
        print(f"{i:>3} {day:>3} {pid:>9} {eng[0]:>7} {eng[1]:>7} | {ph[0]:>7} {ph[1]:>7} | {d0:>+5} {d1:>+5} {len(pin):>3}/{len(multi):>3}    {flag} {kinds}")
        if state.done:
            print("done"); break
    return 0


if __name__ == "__main__":
    sys.exit(main())
