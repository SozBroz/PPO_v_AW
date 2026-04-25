#!/usr/bin/env python3
"""Phase 11J-FUNDS-DEEP — per-envelope funds drift tracer.

For a target gid, replay all envelopes and at every envelope boundary
record (engine_funds[opp], php_funds[opp_awbw_pid]) and the actions
inside the envelope. Print only the envelopes where drift CHANGES
between consecutive turns, so we can tag which action introduced it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

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
# Phase 11J-L1-BUILD-FUNDS-SHIP — extra catalogs added so the 25
# BUILD-FUNDS-RESIDUAL gids that landed via the v2 936 audit (which scrapes
# beyond the std 800) are also traceable here.
_EXTRA_CATALOGS = (
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
    ROOT / "data" / "amarriner_gl_colin_batch.json",
)
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _summarize_actions(actions):
    """Return a Counter-string of action kinds in this envelope."""
    counts = defaultdict(int)
    details = []
    for obj in actions:
        if isinstance(obj, dict):
            kind = obj.get("action") or obj.get("type") or "?"
            counts[kind] += 1
            if kind in ("Build", "Fire", "Capt", "AttackSeam", "Power"):
                details.append((kind, obj))
    return dict(counts), details


def _php_funds_by_pid(frame):
    out = {}
    for pl in (frame.get("players") or {}).values():
        try:
            pid = int(pl.get("id"))
            funds = int(pl.get("funds") or 0)
            out[pid] = funds
        except (TypeError, ValueError):
            continue
    return out


def trace(gid: int) -> int:
    by_id: dict[int, dict] = {}
    for cat_path in (CATALOG, *_EXTRA_CATALOGS):
        if not cat_path.exists():
            continue
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        for g in (cat.get("games") or {}).values():
            if isinstance(g, dict) and "games_id" in g:
                by_id.setdefault(int(g["games_id"]), g)
    meta = by_id[gid]

    random.seed(_seed_for_game(CANONICAL_SEED, gid))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
    zpath = ZIPS / f"{gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    if replay_snapshot_pairing(len(frames), len(envs)) is None:
        print(f"unsupported pairing for gid={gid}")
        return 1
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    print(f"gid={gid} co0={co0} co1={co1} matchup={meta.get('matchup')}")
    print(f"{'env':>4} {'pid':>10} {'day':>4} {'actor':>6} "
          f"{'eng[0]':>7} {'eng[1]':>7} {'php[0]':>7} {'php[1]':>7} "
          f"{'d[0]':>6} {'d[1]':>6} {'D_d[0]':>7} {'D_d[1]':>7} actions")

    prev_d0 = 0
    prev_d1 = 0
    pid_to_engine = awbw_to_engine
    pid0, pid1 = sorted(awbw_to_engine.keys())
    eng_pid_for = {0: engine_to_awbw[0], 1: engine_to_awbw[1]}
    for env_i, (pid, day, actions) in enumerate(envs):
        # Phase 11K-FIRE-FRAC-COUNTER-SHIP — populate the post-envelope HP
        # pin so the override consumer in oracle_zip_replay can recover
        # sub-display-HP counter damage.
        snap_pre = env_i + 1
        if snap_pre < len(frames):
            pin: dict[int, int] = {}
            for u in (frames[snap_pre].get("units") or {}).values():
                try:
                    uid = int(u["id"])
                    hp = float(u["hit_points"])
                except (TypeError, ValueError, KeyError):
                    continue
                pin[uid] = max(0, min(100, int(round(hp * 10))))
            state._oracle_post_envelope_units_by_id = pin
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
            state._oracle_post_envelope_multi_hit_defenders = {
                uid for uid, c in def_hits.items() if c > 1
            }
        else:
            state._oracle_post_envelope_units_by_id = None
            state._oracle_post_envelope_multi_hit_defenders = None
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
        frame_after = frames[snap_i]
        php = _php_funds_by_pid(frame_after)
        eng_funds = [int(state.funds[0]), int(state.funds[1])]
        php_funds = [php.get(eng_pid_for[0], 0), php.get(eng_pid_for[1], 0)]
        d0 = eng_funds[0] - php_funds[0]
        d1 = eng_funds[1] - php_funds[1]
        Dd0 = d0 - prev_d0
        Dd1 = d1 - prev_d1
        actor_e = pid_to_engine.get(pid)
        counts, details = _summarize_actions(actions)
        marker = "*" if (Dd0 != 0 or Dd1 != 0) else " "
        if True:
            print(f"{marker}{env_i:>3} {pid:>10} {day:>4} {actor_e:>6} "
                  f"{eng_funds[0]:>7} {eng_funds[1]:>7} {php_funds[0]:>7} {php_funds[1]:>7} "
                  f"{d0:>6} {d1:>6} {Dd0:>+7} {Dd1:>+7} {counts}")
        prev_d0 = d0
        prev_d1 = d1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    args = ap.parse_args()
    return trace(args.gid)


if __name__ == "__main__":
    raise SystemExit(main())
