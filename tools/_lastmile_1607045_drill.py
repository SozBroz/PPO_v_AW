"""Drill action-by-action through gid 1607045 around env 27 (build no-op)."""
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
    UnsupportedOracleAction,
)
from engine.map_loader import load_map
from engine.game import make_initial_state


def _kind(obj: dict) -> str:
    return str(obj.get("action") or obj.get("type") or obj.get("kind") or "?")


def _summarize(obj: dict) -> str:
    parts = [_kind(obj)]
    for k in ("unitName", "buildUnitName", "fromX", "fromY", "toX", "toY",
              "x", "y", "tileX", "tileY", "playerCash", "cost",
              "coName", "powerName", "playerId"):
        if k in obj:
            parts.append(f"{k}={obj[k]}")
    return " ".join(parts)


def main() -> int:
    gid = 1607045
    co0, co1 = 5, 28
    map_id = 77060
    focus_env = int(sys.argv[1]) if len(sys.argv) > 1 else 27

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
        starting_funds=0, tier_name="T1",
        replay_first_mover=first_mover,
    )

    def _setup_pin(env_i: int, actions: list) -> None:
        post_frame = frames[env_i + 1] if (env_i + 1) < len(frames) else None
        if post_frame is None:
            state._oracle_post_envelope_units_by_id = None
            state._oracle_post_envelope_multi_hit_defenders = None
            return
        pin: dict[int, int] = {}
        for u in (post_frame.get("units") or {}).values():
            try:
                uid = int(u["id"])
                hp = float(u["hit_points"])
            except (TypeError, ValueError, KeyError):
                continue
            pin[uid] = max(0, min(100, int(round(hp * 10))))
        def_hits: dict[int, int] = {}
        for obj in actions:
            if not isinstance(obj, dict):
                continue
            if obj.get("action") not in ("Fire", "AttackSeam"):
                continue
            ci = obj.get("combatInfo")
            if not isinstance(ci, dict):
                continue
            d = ci.get("defender")
            if not isinstance(d, dict):
                continue
            try:
                d_uid = int(d.get("units_id"))
            except (TypeError, ValueError):
                continue
            def_hits[d_uid] = def_hits.get(d_uid, 0) + 1
        multi = {uid for uid, c in def_hits.items() if c > 1}
        state._oracle_post_envelope_units_by_id = pin
        state._oracle_post_envelope_multi_hit_defenders = multi

    for i, (pid, day, actions) in enumerate(envs):
        if i < focus_env:
            _setup_pin(i, actions)
            for obj in actions:
                if state.done:
                    break
                try:
                    apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
                except UnsupportedOracleAction as e:
                    print(f"  pre env={i} ABORT {type(e).__name__}: {e}")
                    return 1
            continue
        _setup_pin(i, actions)

        eng_seat = awbw_to_engine.get(pid)
        print(f"\n=== env {i} day {day} pid={pid} eng_seat={eng_seat} actions={len(actions)} ===")
        print(f"  funds before envelope: P0={state.funds[0]} P1={state.funds[1]}  active_player={state.active_player}")
        for j, obj in enumerate(actions):
            f0, f1 = state.funds[0], state.funds[1]
            ap = state.active_player
            try:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
                f0a, f1a = state.funds[0], state.funds[1]
                d0 = f0a - f0
                d1 = f1a - f1
                tag = ""
                if d0 != 0 or d1 != 0:
                    tag = f" d(P0={d0:+d}, P1={d1:+d})"
                print(f"  [{j:>3}] AP={ap} eng[{f0a:>6},{f1a:>6}] {_summarize(obj)}{tag}")
            except UnsupportedOracleAction as e:
                print(f"  [{j:>3}] AP={ap} eng[{f0:>6},{f1:>6}] !!FAIL!! {type(e).__name__}: {e}")
                print(f"        action: {_summarize(obj)}")
                print(f"        raw: {obj}")
                return 0
            except Exception as e:
                print(f"  [{j:>3}] AP={ap} eng[{f0:>6},{f1:>6}] ABORT {type(e).__name__}: {e}")
                return 1

        if i >= focus_env:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
