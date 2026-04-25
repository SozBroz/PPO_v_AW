"""Trace silent skips through the replay (capture all UnsupportedOracleAction-suppressed events)."""

from __future__ import annotations

import json
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

    n_act = 0
    for env_idx, (pid, day, actions) in enumerate(envs):
        for ai, obj in enumerate(actions):
            if state.done:
                break
            kind = obj.get("action")
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    before_engine_step=None,
                    envelope_awbw_player_id=pid,
                )
                n_act += 1
            except UnsupportedOracleAction as e:
                msg = str(e)
                if "Build no-op" in msg or "Move: mover not found" in msg:
                    print(f"\nFAIL env={env_idx} ai={ai} day={day} kind={kind} acts={n_act}: {e}")
                    nu = obj.get("newUnit")
                    if isinstance(nu, dict):
                        g = nu.get("global") or next(iter(nu.values()), {})
                        if isinstance(g, dict):
                            print(f"  Build target: id={g.get('units_id')} type={g.get('units_name')} pos=(y={g.get('units_y')}, x={g.get('units_x')}) player={g.get('units_players_id')}")
                    return
                else:
                    print(f"FAIL env={env_idx} ai={ai} day={day}: {e}")
                    return


if __name__ == "__main__":
    main()
