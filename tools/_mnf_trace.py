"""Trace BLACK_BOAT roster across all days for gid 1626236 to find when the boat disappears."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from engine.unit import UnitType
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
    target_type_name = sys.argv[2]
    target_type = getattr(UnitType, target_type_name)

    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
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

    def _snapshot_count():
        out = {}
        for seat, lst in state.units.items():
            out[seat] = [(u.unit_id, u.pos, u.hp, u.is_alive) for u in lst if u.unit_type == target_type]
        return out

    print(f"INITIAL {target_type_name}: {_snapshot_count()}")
    n_act = 0
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
                n_act += 1
            except UnsupportedOracleAction as e:
                print(f"\nFAIL env={env_idx} ai={ai} day={day}: {e}")
                print(f"  {target_type_name} state: {_snapshot_count()}")
                return
        snap = _snapshot_count()
        print(f"after env={env_idx} pid={pid} day={day} actions_total={n_act}: P0={snap.get(0, [])} P1={snap.get(1, [])}")


if __name__ == "__main__":
    main()
