"""Walk gid=1635846 envelope-by-envelope; print engine vs PHP funds for both seats."""
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
    gid = int(sys.argv[1]) if len(sys.argv) > 1 else 1635846
    co0 = int(sys.argv[2]) if len(sys.argv) > 2 else 12   # Hawke
    co1 = int(sys.argv[3]) if len(sys.argv) > 3 else 8    # Sensei (per PHP)
    map_id = int(sys.argv[4]) if len(sys.argv) > 4 else 123858

    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    map_pool = REPO / "data/gl_map_pool.json"
    maps_dir = REPO / "data/maps"

    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    map_data = load_map(map_id, map_pool, maps_dir)
    state = make_initial_state(
        map_data, co0, co1,
        starting_funds=0,
        tier_name="T1",
        replay_first_mover=first_mover,
    )

    # PHP funds per frame
    def php_funds(frame: dict) -> tuple[int, int]:
        # frame['players'] keyed by index, players_funds field
        players = frame.get("players", {})
        out = [0, 0]
        for k, p in players.items() if isinstance(players, dict) else []:
            try:
                pid = int(p.get("id"))
                eng = awbw_to_engine.get(pid)
                if eng is None:
                    continue
                out[eng] = int(p.get("funds", 0))
            except Exception:
                continue
        return tuple(out)

    print(f"gid={gid} co0={co0} co1={co1} envs={len(envs)} frames={len(frames)} first_mover={first_mover}")
    print(f"awbw_to_engine={awbw_to_engine}")
    print()
    print(f"{'env':>3} {'day':>3} {'pid':>9} {'eng_p0':>7} {'eng_p1':>7} | {'php_p0':>7} {'php_p1':>7} | {'d_p0':>5} {'d_p1':>5}  notes")

    for i, (pid, day, actions) in enumerate(envs):
        # Apply this envelope
        for obj in actions:
            if state.done:
                break
            try:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
            except Exception as e:
                print(f"  env={i} ABORT {type(e).__name__}: {e}")
                return 1
        # Snapshot
        eng_funds = (state.funds[0], state.funds[1])
        php_idx = i + 1 if (i + 1) < len(frames) else i
        ph = php_funds(frames[php_idx])
        d0 = eng_funds[0] - ph[0]
        d1 = eng_funds[1] - ph[1]
        kinds = ",".join(sorted({(o.get("action") or "?") for o in actions if isinstance(o, dict)}))[:32]
        flag = "    "
        if abs(d0) >= 100 or abs(d1) >= 100:
            flag = "!!!!"
        print(f"{i:>3} {day:>3} {pid:>9} {eng_funds[0]:>7} {eng_funds[1]:>7} | {ph[0]:>7} {ph[1]:>7} | {d0:>+5} {d1:>+5}  {flag} {kinds}")
        if state.done:
            print("  state.done")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
