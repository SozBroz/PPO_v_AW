#!/usr/bin/env python3
"""Phase 11J-F2-KOAL-FU-ORACLE-FUNDS — per-gid funds drill.

For each requested gid, replays the AWBW oracle pipeline and compares the
engine's per-player funds vs the PHP snapshot's per-player funds at every
``p:`` envelope boundary. Stops at the first ``oracle_gap`` (Build no-op
typically) and dumps:

  * funds delta at the failing action,
  * the day boundary where the delta first appears,
  * per-day delta evolution,
  * active CO ids and per-player property counts (income vs comm tower vs lab),
  * the catalog row's COs (sanity).

Usage:
  python tools/_phase11j_funds_drill.py --gid 1621434 --gid 1621898 ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

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

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
CATALOG_EXTRAS = ROOT / "data" / "amarriner_gl_extras_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _frame_funds_by_engine(frame: dict, awbw_to_engine: dict[int, int]) -> dict[int, int]:
    out = {0: 0, 1: 0}
    for k, pl in (frame.get("players") or {}).items():
        try:
            pid = int(pl.get("id"))
        except (TypeError, ValueError):
            continue
        if pid in awbw_to_engine:
            try:
                out[int(awbw_to_engine[pid])] = int(pl.get("funds") or 0)
            except (TypeError, ValueError):
                pass
    return out


def _frame_meta(frame: dict) -> dict[str, Any]:
    return {
        "day": frame.get("day"),
        "active_player_id": frame.get("active_player_id"),
        "turn": frame.get("turn"),
    }


def _engine_prop_counts(state: GameState, player: int) -> dict[str, int]:
    income = sum(
        1 for p in state.properties
        if p.owner == player and not p.is_comm_tower and not p.is_lab
    )
    towers = sum(1 for p in state.properties if p.owner == player and p.is_comm_tower)
    labs = sum(1 for p in state.properties if p.owner == player and p.is_lab)
    return {"income": income, "towers": towers, "labs": labs}


def _frame_prop_counts(frame: dict, awbw_to_engine: dict[int, int]) -> dict[int, dict[str, int]]:
    """Property ownership per engine player from PHP frame ``buildings`` dict.

    ``buildings_team`` is the AWBW player id of the owner (string PHP).
    ``terrain_id`` decides income/tower/lab category — same engine terrain table.
    """
    from engine.terrain import get_terrain
    out = {0: {"income": 0, "towers": 0, "labs": 0},
           1: {"income": 0, "towers": 0, "labs": 0}}
    for b in (frame.get("buildings") or {}).values():
        team_raw = b.get("buildings_team")
        try:
            team = int(team_raw) if team_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            team = None
        if team is None or team not in awbw_to_engine:
            continue
        eng = int(awbw_to_engine[team])
        tid_raw = b.get("terrain_id") or b.get("buildings_terrain_id") or b.get("type")
        try:
            tid = int(tid_raw)
        except (TypeError, ValueError):
            continue
        info = get_terrain(tid)
        if info is None or not info.is_property:
            continue
        if info.is_comm_tower:
            out[eng]["towers"] += 1
        elif info.is_lab:
            out[eng]["labs"] += 1
        else:
            out[eng]["income"] += 1
    return out


def drill_one(gid: int, max_envelopes: Optional[int] = None,
              verbose_from_day: Optional[int] = None) -> dict[str, Any]:
    by_id: dict[int, Any] = {}
    for cat_path in (CATALOG, CATALOG_EXTRAS):
        if not cat_path.exists():
            continue
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        for g in (cat.get("games") or {}).values():
            if isinstance(g, dict) and "games_id" in g:
                by_id[int(g["games_id"])] = g
    meta = by_id[gid]

    random.seed(_seed_for_game(CANONICAL_SEED, gid))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)

    zpath = ZIPS / f"{gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    out: dict[str, Any] = {
        "gid": gid,
        "co_p0": co0, "co_p1": co1,
        "matchup": meta.get("matchup"),
        "map_id": meta.get("map_id"),
        "tier": meta.get("tier"),
        "n_envelopes": len(envs),
        "n_frames": len(frames),
        "awbw_to_engine": awbw_to_engine,
        "per_envelope": [],
        "result": None,
    }

    last_day_seen = None
    for env_i, (pid, day, actions) in enumerate(envs):
        if max_envelopes is not None and env_i >= max_envelopes:
            break
        # Frame index follows envelopes (frames[0] is initial; frames[i+1] is post-envelope-i).
        frame_after = frames[env_i + 1] if env_i + 1 < len(frames) else None

        try:
            for j, obj in enumerate(actions):
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    envelope_awbw_player_id=pid,
                )
        except UnsupportedOracleAction as e:
            row = {
                "env_i": env_i, "pid": pid, "day": day,
                "n_actions_in_env": len(actions),
                "fail_at_action_idx": j,
                "fail_action_kind": obj.get("action"),
                "fail_msg": str(e),
                "engine_funds": dict(zip([0, 1], [int(state.funds[0]), int(state.funds[1])])),
                "engine_props": {
                    0: _engine_prop_counts(state, 0),
                    1: _engine_prop_counts(state, 1),
                },
            }
            if frame_after is not None:
                row["php_funds_post_env"] = _frame_funds_by_engine(frame_after, awbw_to_engine)
                row["php_props_post_env"] = _frame_prop_counts(frame_after, awbw_to_engine)
            # Also pre-envelope frame for fund baseline
            if env_i < len(frames):
                row["php_funds_pre_env"] = _frame_funds_by_engine(frames[env_i], awbw_to_engine)
            out["per_envelope"].append(row)
            out["result"] = "oracle_gap_at_failure"
            return out
        except Exception as e:
            out["per_envelope"].append({
                "env_i": env_i, "pid": pid, "day": day,
                "fatal": f"{type(e).__name__}: {e}",
            })
            out["result"] = "fatal_exception"
            return out

        # Compare engine vs frame funds AFTER this envelope is applied.
        eng_funds = {0: int(state.funds[0]), 1: int(state.funds[1])}
        eng_props = {0: _engine_prop_counts(state, 0), 1: _engine_prop_counts(state, 1)}
        php_funds = _frame_funds_by_engine(frame_after, awbw_to_engine) if frame_after else None
        php_props = _frame_prop_counts(frame_after, awbw_to_engine) if frame_after else None

        delta = None
        if php_funds is not None:
            delta = {p: eng_funds[p] - php_funds[p] for p in (0, 1)}

        new_day = (last_day_seen != day)
        force = (verbose_from_day is not None and day is not None
                 and int(day) >= int(verbose_from_day))
        # Only print when day changes, when delta != 0, or first/last few rows.
        if new_day or (delta and any(v != 0 for v in delta.values())) or force:
            out["per_envelope"].append({
                "env_i": env_i, "pid": pid, "day": day, "new_day": new_day,
                "n_actions": len(actions),
                "engine_funds": eng_funds,
                "php_funds": php_funds,
                "delta_engine_minus_php": delta,
                "engine_props": eng_props,
                "php_props": php_props,
            })
        last_day_seen = day

    out["result"] = "completed"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gid", type=int, action="append", required=True)
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "logs" / "phase11j_funds_drill.json")
    ap.add_argument("--max-envelopes", type=int, default=None)
    ap.add_argument("--verbose-from-day", type=int, default=None,
                    help="Log every envelope from this day onward.")
    args = ap.parse_args()

    cases: list[dict[str, Any]] = []
    for gid in args.gid:
        try:
            cases.append(drill_one(gid, args.max_envelopes,
                                   verbose_from_day=args.verbose_from_day))
        except Exception as e:
            cases.append({"gid": gid, "drill_exception": f"{type(e).__name__}: {e}"})

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"cases": cases}, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    # Summary print
    for c in cases:
        gid = c["gid"]
        print(f"\n===== gid {gid} | {c.get('matchup')} | "
              f"co_p0={c.get('co_p0')} co_p1={c.get('co_p1')} | map={c.get('map_id')} "
              f"tier={c.get('tier')} =====")
        print(f"  result={c.get('result')}")
        if "drill_exception" in c:
            print("  EXC:", c["drill_exception"])
            continue
        rows = c.get("per_envelope") or []
        # Show first few + last few
        for r in rows[:6]:
            print(" ", r)
        if len(rows) > 12:
            print("   ... (omitted middle) ...")
        for r in rows[-12:]:
            print(" ", r)

    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
