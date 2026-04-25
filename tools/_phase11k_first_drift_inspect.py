#!/usr/bin/env python3
"""Phase 11K-FIRST-DRIFT-INSPECT — for gid 1635679 dump every action in
envelope 10 (P0 day 6) and the resulting (3,4) unit HP at envelope-end
in BOTH engine and PHP. The state-mismatch diagnostic flagged this as
the first HP divergence (engine=100 / PHP=97 / delta=3 internal).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.replay_snapshot_compare import replay_snapshot_pairing

CATS = [
    ROOT / "data" / "amarriner_gl_std_catalog.json",
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
]
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, default=1635679)
    ap.add_argument("--env", type=int, default=10)
    ap.add_argument("--row", type=int, default=3)
    ap.add_argument("--col", type=int, default=4)
    ap.add_argument("--player", type=int, default=0,
                    help="Engine player whose unit to track")
    args = ap.parse_args()

    by_id = {}
    for cp in CATS:
        if not cp.exists():
            continue
        cat = json.loads(cp.read_text(encoding="utf-8"))
        for g in (cat.get("games") or {}).values():
            if isinstance(g, dict) and "games_id" in g:
                by_id[int(g["games_id"])] = g
    meta = by_id[args.gid]

    random.seed(_seed_for_game(CANONICAL_SEED, args.gid))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    if replay_snapshot_pairing(len(frames), len(envs)) is None:
        print("unsupported pairing")
        return 1
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    print(f"gid={args.gid} co_p0={co0} co_p1={co1} engine_to_awbw={engine_to_awbw}")
    print(f"target=(p={args.player},r={args.row},c={args.col}) first_drift_env={args.env}\n")

    for env_i, (pid, day, actions) in enumerate(envs):
        if env_i > args.env:
            break
        for ai, obj in enumerate(actions):
            kind = obj.get("action") if isinstance(obj, dict) else None
            apply_oracle_action_json(state, obj, awbw_to_engine,
                                     envelope_awbw_player_id=pid)
            if env_i == args.env:
                u = next(
                    (uu for uu in state.units[args.player]
                     if uu.pos == (args.row, args.col)),
                    None,
                )
                hp_str = f"hp={u.hp}({u.unit_type.name})" if u else "absent"
                # Compact action snippet
                parts = {k: obj.get(k) for k in
                         ("action", "src", "dst", "from", "to", "fromX",
                          "fromY", "toX", "toY", "fromTile", "toTile",
                          "combatInfo", "attacker", "defender", "type")
                         if k in obj}
                print(f"  env={env_i} ai={ai} kind={kind} target_{args.player}_{args.row}_{args.col} {hp_str}")
                # If this action carries combatInfo at the target, dump it.
                ci = obj.get("combatInfo") if isinstance(obj, dict) else None
                if ci:
                    for side in ("attacker", "defender"):
                        d = ci.get(side) or {}
                        try:
                            x = int(d.get("x"))
                            y = int(d.get("y"))
                        except (TypeError, ValueError):
                            continue
                        if (y, x) == (args.row, args.col):
                            print(f"      combatInfo.{side} pos=({y},{x}) "
                                  f"hp={d.get('hit_points')} "
                                  f"hp_internal={d.get('hpInternal')} "
                                  f"u_id={d.get('units_id')}")

        # End of envelope: dump engine vs PHP target unit
        if env_i == args.env:
            u = next(
                (uu for uu in state.units[args.player]
                 if uu.pos == (args.row, args.col)),
                None,
            )
            print()
            print(f"=== END env {env_i} day {day} ===")
            if u:
                print(f"  ENGINE p{args.player} ({args.row},{args.col}) "
                      f"{u.unit_type.name} hp={u.hp} fuel={u.fuel} ammo={u.ammo}")
            else:
                print(f"  ENGINE: no unit at ({args.row},{args.col}) for p{args.player}")

            fa = frames[env_i + 1] if env_i + 1 < len(frames) else None
            if fa:
                target_awbw_pid = engine_to_awbw[args.player]
                for uu in (fa.get("units") or {}).values():
                    try:
                        pos = (int(uu["y"]), int(uu["x"]))
                        plid = int(uu["players_id"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if pos == (args.row, args.col) and plid == target_awbw_pid:
                        print(f"  PHP    awbw_pid={plid} ({args.row},{args.col}) "
                              f"name={uu.get('name')} hp={uu.get('hit_points')} "
                              f"fuel={uu.get('fuel')} ammo={uu.get('ammo')}")
                        break
                else:
                    print(f"  PHP: no unit at ({args.row},{args.col}) for awbw_pid={target_awbw_pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
