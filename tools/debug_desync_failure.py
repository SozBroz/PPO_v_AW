"""
Print engine state at the first oracle / engine exception when replaying a zip.

Usage (repo root)::

  python tools/debug_desync_failure.py --games-id 1619191
  python tools/debug_desync_failure.py --games-id 1620188 --max-actions 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import ActionStage, get_legal_actions  # noqa: E402
from engine.game import make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from tools.amarriner_catalog_cos import pair_catalog_cos_ids  # noqa: E402
from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)


def _dump_state(label: str, state) -> None:
    la = get_legal_actions(state)
    su = state.selected_unit
    print(f"--- {label} ---")
    print(
        f"  active_player={state.active_player} stage={state.action_stage.name} "
        f"day~{state.turn} sel={su and su.unit_type.name}@{su and su.pos} "
        f"mpos={state.selected_move_pos}"
    )
    print(f"  legal_actions: {len(la)} -> {[a.action_type.name for a in la[:16]]}")
    alive = sum(1 for p in (0, 1) for u in state.units[p] if u.is_alive)
    print(f"  alive_units_total={alive}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-id", type=int, required=True)
    ap.add_argument("--max-actions", type=int, default=None)
    args = ap.parse_args()

    gid = args.games_id
    cat_path = ROOT / "data" / "amarriner_gl_std_catalog.json"
    catalog = json.loads(cat_path.read_text(encoding="utf-8"))
    meta = None
    for _k, g in (catalog.get("games") or {}).items():
        if isinstance(g, dict) and int(g.get("games_id", 0)) == gid:
            meta = g
            break
    if meta is None:
        print(f"no catalog row for games_id={gid}", file=sys.stderr)
        return 1

    zpath = ROOT / "replays" / "amarriner_gl" / f"{gid}.zip"
    if not zpath.is_file():
        print(f"missing zip {zpath}", file=sys.stderr)
        return 1

    co0, co1 = pair_catalog_cos_ids(meta)
    frames = load_replay(zpath)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    map_data = load_map(
        int(meta["map_id"]),
        ROOT / "data" / "gl_map_pool.json",
        ROOT / "data" / "maps",
    )
    envs = parse_p_envelopes_from_zip(zpath)
    if not envs:
        print(
            "no a<game_id> action stream in zip (ReplayVersion 1 snapshot-only); "
            "nothing to replay.",
            file=sys.stderr,
        )
        return 1
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data,
        co0,
        co1,
        starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    n = 0
    try:
        for env_i, (_pid, day, actions) in enumerate(envs):
            for j, obj in enumerate(actions):
                if args.max_actions is not None and n >= args.max_actions:
                    print(f"Stopped after --max-actions={args.max_actions} (no exception).")
                    return 0
                kind = obj.get("action")
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=_pid)
                n += 1
                if state.done:
                    print(f"Replay finished ok after {n} actions.")
                    return 0
    except Exception as exc:
        print(f"\nEXCEPTION after {n} actions: {type(exc).__name__}: {exc}")
        print(f"Last envelope index hint: env_i={env_i} j={j} day={day} kind={kind}")
        _dump_state("at exception", state)
        if state.action_stage == ActionStage.ACTION:
            print(
                "  hint: legal=[] in ACTION usually means selected_move_pos is None, "
                "or boarding branch returned no LOAD/JOIN (see engine/action.py "
                "_get_action_actions)."
            )
        return 0

    print(f"No exception in replay ({n} actions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
