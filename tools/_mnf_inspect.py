"""
Phase 11J-FINAL — mover-not-found probe.

For a given games_id, run the oracle replay until it raises and dump:
  - the failing AWBW action JSON (with paths.global, units_id, units_hit_points)
  - the engine roster snapshot at that moment
  - whether the AWBW units_id matches any unit_id (alive or dead) on the map
  - whether any same-type unit exists on the engine map for either seat
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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


def _catalog_lookup(gid: int) -> dict[str, Any]:
    for cat in (
        Path("data/amarriner_gl_std_catalog.json"),
        Path("data/amarriner_gl_extras_catalog.json"),
    ):
        if not cat.exists():
            continue
        data = json.loads(cat.read_text(encoding="utf-8"))
        games = data.get("games", data) if isinstance(data, dict) else data
        if isinstance(games, dict):
            row = games.get(str(gid)) or games.get(gid)
            if row:
                return row
            for r in games.values():
                if int(r.get("games_id", 0)) == gid:
                    return r
        else:
            for r in games:
                if int(r.get("games_id", 0)) == gid:
                    return r
    raise SystemExit(f"gid {gid} not found in catalogs")


def _summarize_roster(state) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seat, lst in state.units.items():
        out[f"P{seat}"] = [
            {
                "id": int(u.unit_id),
                "type": u.unit_type.name,
                "pos": u.pos,
                "hp": u.hp,
                "alive": u.is_alive,
                "moved": u.moved,
                "player": int(u.player),
            }
            for u in lst
        ]
    return out


def main() -> None:
    gid = int(sys.argv[1])
    row = _catalog_lookup(gid)
    zip_path = Path(f"replays/amarriner_gl/{gid}.zip")
    map_id = int(row["map_id"])
    co0 = int(row["co_p0_id"])
    co1 = int(row["co_p1_id"])
    tier = str(row["tier"])

    frames = load_replay(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    map_data = load_map(map_id, Path("data/gl_map_pool.json"), Path("data/maps"))
    envs = parse_p_envelopes_from_zip(zip_path)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data,
        co0,
        co1,
        starting_funds=0,
        tier_name=tier,
        replay_first_mover=first_mover,
    )

    n_env = 0
    n_act = 0
    last_ok_action: dict[str, Any] | None = None
    failing_obj: dict[str, Any] | None = None
    failing_pid: int | None = None
    failing_day: int | None = None
    for env_idx, (pid, day, actions) in enumerate(envs):
        for ai, obj in enumerate(actions):
            if state.done:
                break
            try:
                apply_oracle_action_json(
                    state,
                    obj,
                    awbw_to_engine,
                    before_engine_step=None,
                    envelope_awbw_player_id=pid,
                )
                n_act += 1
                last_ok_action = obj
            except UnsupportedOracleAction as e:
                failing_obj = obj
                failing_pid = pid
                failing_day = day
                print(f"[FAIL] env={env_idx} action_idx={ai} pid={pid} day={day}: {e}")
                break
        if failing_obj is not None:
            break
        n_env += 1

    if failing_obj is None:
        print("REPLAY COMPLETED WITHOUT EXCEPTION")
        return

    print("\n=== FAILING ACTION ===")
    print(json.dumps(failing_obj, indent=2, default=str)[:4000])

    inner = failing_obj.get("Move") or failing_obj
    paths = inner.get("paths") or {}
    glb = paths.get("global") if isinstance(paths, dict) else None
    unit = inner.get("unit") or {}
    aw_uid = unit.get("units_id")
    aw_type = unit.get("units_name")
    aw_player = unit.get("units_players_id")
    aw_hp = unit.get("units_hit_points")

    print("\n=== KEY FIELDS ===")
    print(f"units_id: {aw_uid}")
    print(f"units_name: {aw_type}")
    print(f"units_players_id: {aw_player}")
    print(f"units_hit_points: {aw_hp!r}")
    print(f"paths.global len: {len(glb) if isinstance(glb, list) else 'n/a'} -> {glb}")
    print(f"awbw->engine: {awbw_to_engine}")
    eng_seat = awbw_to_engine.get(int(aw_player)) if aw_player is not None else None
    print(f"derived engine seat: {eng_seat}")
    print(f"state.active_player: {state.active_player}")

    print("\n=== ROSTER (alive only) ===")
    for seat, lst in state.units.items():
        alive = [u for u in lst if u.is_alive]
        print(f"P{seat} ({len(alive)} alive):")
        for u in alive:
            mark = ""
            try:
                if aw_uid is not None and int(u.unit_id) == int(aw_uid):
                    mark = "  <-- ID MATCH"
            except (TypeError, ValueError):
                pass
            print(
                f"  id={u.unit_id} type={u.unit_type.name} pos={u.pos} "
                f"hp={u.hp} moved={u.moved}{mark}"
            )

    print("\n=== DEAD ROSTER ===")
    for seat, lst in state.units.items():
        dead = [u for u in lst if not u.is_alive]
        if dead:
            print(f"P{seat} ({len(dead)} dead):")
            for u in dead:
                mark = ""
                try:
                    if aw_uid is not None and int(u.unit_id) == int(aw_uid):
                        mark = "  <-- ID MATCH (DEAD)"
                except (TypeError, ValueError):
                    pass
                print(
                    f"  id={u.unit_id} type={u.unit_type.name} pos={u.pos}{mark}"
                )

    print("\n=== ID PRESENCE ON MAP ===")
    found_alive: list[tuple[int, Any]] = []
    found_dead: list[tuple[int, Any]] = []
    for seat, lst in state.units.items():
        for u in lst:
            try:
                if aw_uid is not None and int(u.unit_id) == int(aw_uid):
                    if u.is_alive:
                        found_alive.append((seat, u))
                    else:
                        found_dead.append((seat, u))
            except (TypeError, ValueError):
                pass
    print(f"id={aw_uid} alive={len(found_alive)} dead={len(found_dead)}")

    print("\n=== TYPE PRESENCE ON MAP ===")
    if aw_type:
        from tools.oracle_zip_replay import _name_to_unit_type

        try:
            want_t = _name_to_unit_type(str(aw_type))
        except Exception as e:
            print(f"name->type failed: {e}")
            want_t = None
        if want_t is not None:
            for seat, lst in state.units.items():
                same = [u for u in lst if u.is_alive and u.unit_type == want_t]
                print(f"P{seat} alive {want_t.name} count={len(same)}")
                for u in same:
                    print(
                        f"  id={u.unit_id} pos={u.pos} hp={u.hp} moved={u.moved}"
                    )

    print(f"\n=== STATS env_idx={env_idx} action_idx={ai} ===")
    print(f"envelopes_applied: {n_env}, actions_applied: {n_act}")


if __name__ == "__main__":
    main()
