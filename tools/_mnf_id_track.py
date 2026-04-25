"""Track engine position/HP of a specific unit_id through every action of a target envelope."""

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


def _find(state, target_id: int):
    for seat, lst in state.units.items():
        for u in lst:
            if int(u.unit_id) == target_id:
                return (seat, u.pos, u.hp, u.is_alive, u.unit_type.name)
    return None


def main() -> None:
    gid = int(sys.argv[1])
    target_env = int(sys.argv[2])
    target_id = int(sys.argv[3])
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
                return
            if env_idx == target_env:
                kind = obj.get("action") or "?"
                inner = obj.get(kind) or obj
                unit = inner.get("unit") if isinstance(inner, dict) else None
                glb = (unit or {}).get("global") if isinstance(unit, dict) else None
                aw_uid = glb.get("units_id") if isinstance(glb, dict) else None
                aw_name = glb.get("units_name") if isinstance(glb, dict) else None
                aw_x = glb.get("units_x") if isinstance(glb, dict) else None
                aw_y = glb.get("units_y") if isinstance(glb, dict) else None
                paths = inner.get("paths") if isinstance(inner, dict) else None
                pgcoords = ""
                if isinstance(paths, dict) and isinstance(paths.get("global"), list) and paths["global"]:
                    pg = paths["global"]
                    pgcoords = f" path:{pg[0].get('x'),pg[0].get('y')}->{pg[-1].get('x'),pg[-1].get('y')}"
                info = _find(state, target_id)
                print(
                    f"  env={env_idx} ai={ai} {kind} aw_uid={aw_uid} aw_name={aw_name} aw_xy=({aw_x},{aw_y}){pgcoords}\n"
                    f"     -> id={target_id} {info}"
                )
        if env_idx >= target_env:
            return


if __name__ == "__main__":
    main()
