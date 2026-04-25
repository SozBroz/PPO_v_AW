#!/usr/bin/env python3
"""Phase 11J-CLUSTER-B-SHIP — per-envelope HP tracker for specific opponent positions.

Tracks engine HP vs PHP HP for a fixed set of positions across all envelopes
and prints rows where the divergence flips.
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
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.replay_snapshot_compare import replay_snapshot_pairing

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--player", type=int, required=True,
                    help="engine player to track")
    ap.add_argument("--positions", type=str, required=True,
                    help='JSON list of [r, c] positions to track')
    args = ap.parse_args()

    positions = [tuple(p) for p in json.loads(args.positions)]
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id = {int(g["games_id"]): g for g in games.values()
             if isinstance(g, dict) and "games_id" in g}
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
    awbw_pid = engine_to_awbw[args.player]

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    def php_hp(frame, pos):
        for u in (frame.get("units") or {}).values():
            try:
                if int(u["players_id"]) != awbw_pid:
                    continue
                if (int(u["y"]), int(u["x"])) != pos:
                    continue
            except (TypeError, ValueError, KeyError):
                continue
            try:
                return round(float(u["hit_points"]) * 10)
            except (TypeError, ValueError):
                return None
        return None

    def engine_hp(pos):
        u = state.get_unit_at(*pos)
        if u is None or not u.is_alive or int(u.player) != args.player:
            return None
        return int(u.hp)

    header = "env " + " ".join([f"{str(p):>10}" for p in positions])
    print(header)
    prev_eng = {p: None for p in positions}
    prev_php = {p: None for p in positions}
    print(f"--- engine ---")
    for env_i, (pid, day, actions) in enumerate(envs):
        try:
            for obj in actions:
                apply_oracle_action_json(state, obj, awbw_to_engine,
                                         envelope_awbw_player_id=pid)
        except UnsupportedOracleAction as e:
            print(f"oracle_gap@{env_i}: {e}")
            return 1
        snap_i = env_i + 1
        if snap_i >= len(frames):
            break
        eng_vals = {p: engine_hp(p) for p in positions}
        php_vals = {p: php_hp(frames[snap_i], p) for p in positions}
        flips = []
        for p in positions:
            ev, pv = eng_vals[p], php_vals[p]
            pe, pp = prev_eng[p], prev_php[p]
            if (ev != pe) or (pv != pp):
                flips.append((p, pe, pp, ev, pv))
        if flips:
            print(f"env {env_i:>3} actor=P{awbw_to_engine[pid]} day={day}:")
            for (p, pe, pp, ev, pv) in flips:
                marker = " " if ev == pv else "*"
                print(f"  {marker} {str(p):>10} eng {pe}->{ev}  php {pp}->{pv}")
        for p in positions:
            prev_eng[p] = eng_vals[p]
            prev_php[p] = php_vals[p]
        if env_i >= 28:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
