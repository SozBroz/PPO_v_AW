"""Print full roster (all types) at a target envelope, seeded like desync_audit."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

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
        data = json.loads(cat.read_text(encoding="utf-8"))
        games = data.get("games", data) if isinstance(data, dict) else data
        if isinstance(games, dict):
            for r in games.values():
                if int(r.get("games_id", 0)) == gid:
                    return r
    raise SystemExit("missing")


def main() -> None:
    gid = int(sys.argv[1])
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    random.seed(((seed & 0xFFFFFFFF) << 32) | (gid & 0xFFFFFFFF))

    row = _catalog_lookup(gid)
    zip_path = Path(f"replays/amarriner_gl/{gid}.zip")
    frames = load_replay(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(
        frames[0], int(row["co_p0_id"]), int(row["co_p1_id"])
    )
    map_data = load_map(int(row["map_id"]), Path("data/gl_map_pool.json"), Path("data/maps"))
    envs = parse_p_envelopes_from_zip(zip_path)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data, int(row["co_p0_id"]), int(row["co_p1_id"]),
        starting_funds=0, tier_name=str(row["tier"]),
        replay_first_mover=first_mover,
    )

    def _dump(env_i, day, pid):
        print(f"\n--- after env={env_i} day={day} pid={pid} funds={getattr(state,'funds',None)} ---")
        for seat, lst in state.units.items():
            print(f"  P{seat} ({sum(1 for u in lst if u.is_alive)} alive):")
            for u in sorted(lst, key=lambda x: x.unit_id):
                if u.is_alive:
                    print(f"    id={u.unit_id:3d} {u.unit_type.name:10s} pos={u.pos} hp={u.hp} fuel={u.fuel} ammo={u.ammo} moved={u.moved}")

    for env_idx, (pid, day, actions) in enumerate(envs):
        for ai, obj in enumerate(actions):
            if state.done:
                break
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    before_engine_step=None,
                    envelope_awbw_player_id=pid,
                )
            except UnsupportedOracleAction as e:
                print(f"FAIL env={env_idx} ai={ai}: {e}")
                _dump(env_idx, day, pid)
                return
        if start <= env_idx <= end:
            _dump(env_idx, day, pid)
        if env_idx > end:
            return


if __name__ == "__main__":
    main()
