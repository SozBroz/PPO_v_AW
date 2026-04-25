"""Replay envelopes and print every Move action in env=32 day 17 (or another env) along with applied result."""

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
    target_env = int(sys.argv[2])

    # Mirror desync_audit's per-game RNG seed (default --seed=0 in audit).
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

    for env_idx, (pid, day, actions) in enumerate(envs):
        if env_idx == target_env:
            print(f"\n*** env={env_idx} pid={pid} day={day} actions={len(actions)} ***")
            for ai, obj in enumerate(actions):
                kind = obj.get("action") or "?"
                inner = obj.get(kind) or obj
                unit = inner.get("unit") if isinstance(inner, dict) else None
                glb = (unit or {}).get("global") if isinstance(unit, dict) else None
                pre = ""
                if isinstance(glb, dict):
                    pre = (
                        f" uid={glb.get('units_id')} name={glb.get('units_name')} "
                        f"x={glb.get('units_x')} y={glb.get('units_y')} hp={glb.get('units_hit_points')}"
                    )
                paths = inner.get("paths") if isinstance(inner, dict) else None
                ppath = ""
                if isinstance(paths, dict) and isinstance(paths.get("global"), list):
                    pg = paths["global"]
                    ppath = f" path[{len(pg)}]={pg[0]}->{pg[-1]}" if pg else ""
                try:
                    apply_oracle_action_json(
                        state, obj, awbw_to_engine,
                        before_engine_step=None,
                        envelope_awbw_player_id=pid,
                    )
                    print(f"  ai={ai} OK  {kind}{pre}{ppath}")
                except UnsupportedOracleAction as e:
                    print(f"  ai={ai} FAIL {kind}{pre}{ppath}  -> {e}")
                    return
            return
        for ai, obj in enumerate(actions):
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    before_engine_step=None,
                    envelope_awbw_player_id=pid,
                )
            except UnsupportedOracleAction as e:
                print(f"PRE-TARGET FAIL env={env_idx} ai={ai}: {e}")
                return


if __name__ == "__main__":
    main()
