#!/usr/bin/env python3
"""Compare engine vs PHP unit states at consecutive envelopes for a single gid.

Pinpoints which unit's HP / position differs at the moment funds drift first
appears, so we can decide whether the funds delta is downstream of an HP/combat
drift or originates in the income/repair tick itself.
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

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _frame_units(frame: dict, awbw_to_engine: dict[int, int]) -> dict[tuple[int, int], dict]:
    out = {}
    for u in (frame.get("units") or {}).values():
        try:
            x = int(u.get("x"))
            y = int(u.get("y"))
            hp_raw = u.get("hit_points")
            hp = int(round(float(hp_raw) * 10)) if hp_raw is not None else 0
            apid = int(u.get("players_id"))
        except (TypeError, ValueError):
            continue
        if apid not in awbw_to_engine:
            continue
        out[(y, x)] = {
            "name": u.get("name"),
            "hp": hp,
            "player": awbw_to_engine[apid],
            "fuel": u.get("fuel"),
            "ammo": u.get("ammo"),
        }
    return out


def _engine_units(state: GameState) -> dict[tuple[int, int], dict]:
    out = {}
    for p in (0, 1):
        for u in state.units[p]:
            out[u.pos] = {
                "name": u.unit_type.name,
                "hp_internal": u.hp,
                "hp_visual": (u.hp + 9) // 10,
                "player": p,
                "fuel": u.fuel,
                "ammo": u.ammo,
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env-from", type=int, required=True, help="dump at end of this envelope")
    ap.add_argument("--env-to", type=int, required=True, help="dump at end of this envelope (later)")
    ap.add_argument("--player", type=int, default=None, help="filter to this engine player only")
    args = ap.parse_args()

    by_id = {}
    for cat_path in (CATALOG, ROOT / "data" / "amarriner_gl_extras_catalog.json"):
        if cat_path.exists():
            cat = json.loads(cat_path.read_text(encoding="utf-8"))
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
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    snapshots: dict[int, tuple] = {}
    for env_i, (pid, day, actions) in enumerate(envs):
        try:
            for obj in actions:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
        except UnsupportedOracleAction as e:
            print(f"FAIL at env {env_i}: {e}")
            return 1

        if env_i in (args.env_from, args.env_to):
            eng = _engine_units(state)
            php_frame = frames[env_i + 1] if env_i + 1 < len(frames) else None
            php = _frame_units(php_frame, awbw_to_engine) if php_frame else {}
            snapshots[env_i] = (eng, php, dict(zip([0, 1], [int(state.funds[0]), int(state.funds[1])])))

        if env_i > args.env_to:
            break

    for env_i in (args.env_from, args.env_to):
        if env_i not in snapshots:
            continue
        eng, php, funds = snapshots[env_i]
        print(f"\n===== env {env_i} | engine_funds={funds} =====")
        php_funds = {0: 0, 1: 0}
        for k, pl in (frames[env_i + 1].get("players") or {}).items():
            try:
                pid_a = int(pl.get("id"))
                if pid_a in awbw_to_engine:
                    php_funds[awbw_to_engine[pid_a]] = int(pl.get("funds") or 0)
            except (TypeError, ValueError):
                pass
        print(f"        | php_funds={php_funds}")

        # All positions
        all_pos = sorted(set(eng.keys()) | set(php.keys()))
        diffs = []
        for pos in all_pos:
            e = eng.get(pos)
            p = php.get(pos)
            if args.player is not None:
                if (e and e["player"] != args.player) and (p and p["player"] != args.player):
                    continue
            if e is None:
                diffs.append((pos, "PHP_ONLY", p))
            elif p is None:
                diffs.append((pos, "ENGINE_ONLY", e))
            else:
                if e["hp_internal"] != p["hp"]:
                    diffs.append((pos, "HP", {"eng_int": e["hp_internal"], "php_int": p["hp"], "name": e["name"], "player": e["player"]}))
        print(f"  diffs: {len(diffs)}")
        for d in diffs[:80]:
            print(" ", d)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
