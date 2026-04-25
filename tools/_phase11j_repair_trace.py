#!/usr/bin/env python3
"""Phase 11J-F2-FU-FUNDS — instrument _resupply_on_properties / _grant_income.

Replays gid 1621434 and prints, for each call to either method, the player,
funds before/after, units repaired & costs, and the day boundary.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)

import argparse
GID_DEFAULT = 1621434
TARGET_DAY_FROM_DEFAULT = 12

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"

LOG: list[str] = []
CURRENT_DAY = [None]
CURRENT_ENV = [None]
CURRENT_ACTING_PID = [None]


def patch():
    orig_resupply = GameState._resupply_on_properties
    orig_grant = GameState._grant_income

    def patched_resupply(self, player):
        before = int(self.funds[player])
        repairs = []

        # Mirror _resupply_on_properties internals to capture per-unit cost
        co_id = self.co_states[player].co_id
        property_heal = 30 if co_id == 28 else 20
        result_units = []
        for unit in self.units[player]:
            prop = self.get_property_at(*unit.pos)
            if prop is None or prop.owner != player:
                continue
            stats = UNIT_STATS[unit.unit_type]
            cls = stats.unit_class
            is_city = not (
                prop.is_hq or prop.is_lab or prop.is_comm_tower
                or prop.is_base or prop.is_airport or prop.is_port
            )
            qualifies = False
            if cls in ("infantry", "mech", "vehicle", "pipe"):
                qualifies = prop.is_hq or prop.is_base or is_city
            elif cls in ("air", "copter"):
                qualifies = prop.is_airport
            elif cls == "naval":
                qualifies = prop.is_port
            if (
                qualifies and not prop.is_lab and not prop.is_comm_tower
                and unit.hp < 100
            ):
                result_units.append((unit.unit_type.name, unit.pos, unit.hp,
                                     property_heal, prop.terrain_id))

        orig_resupply(self, player)
        after = int(self.funds[player])
        spent = before - after
        LOG.append(
            f"  REPAIR P{player} (env={CURRENT_ENV[0]} day={CURRENT_DAY[0]} "
            f"pid={CURRENT_ACTING_PID[0]}): "
            f"funds {before}->{after} (spent {spent}); eligible={result_units}"
        )

    def patched_grant(self, player):
        before = int(self.funds[player])
        n = self.count_income_properties(player)
        orig_grant(self, player)
        after = int(self.funds[player])
        LOG.append(
            f"  INCOME P{player} (env={CURRENT_ENV[0]} day={CURRENT_DAY[0]} "
            f"pid={CURRENT_ACTING_PID[0]}): "
            f"funds {before}->{after} (+{after - before}; n_income_props={n})"
        )

    GameState._resupply_on_properties = patched_resupply
    GameState._grant_income = patched_grant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, default=GID_DEFAULT)
    ap.add_argument("--from-day", type=int, default=TARGET_DAY_FROM_DEFAULT)
    ap.add_argument("--catalog", type=str, default=str(CATALOG))
    args = ap.parse_args()
    GID = args.gid
    TARGET_DAY_FROM = args.from_day

    patch()
    by_id: dict[int, dict] = {}
    for cat_path_str in (args.catalog,
                         str(ROOT / "data" / "amarriner_gl_extras_catalog.json")):
        cat_path = Path(cat_path_str)
        if not cat_path.exists():
            continue
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        for g in (cat.get("games") or {}).values():
            if isinstance(g, dict) and "games_id" in g:
                by_id[int(g["games_id"])] = g
    meta = by_id[GID]
    random.seed(_seed_for_game(CANONICAL_SEED, GID))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)
    zpath = ZIPS / f"{GID}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )
    for env_i, (pid, day, actions) in enumerate(envs):
        if day is not None and day >= TARGET_DAY_FROM:
            LOG.append(
                f"=== env {env_i} pid={pid} day={day} actions={len(actions)} ==="
            )
        CURRENT_DAY[0] = day
        CURRENT_ENV[0] = env_i
        CURRENT_ACTING_PID[0] = pid
        try:
            for obj in actions:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    envelope_awbw_player_id=pid,
                )
        except Exception as e:
            LOG.append(f"  FAIL: {type(e).__name__}: {e}")
            break
        if day is not None and day >= TARGET_DAY_FROM:
            LOG.append(
                f"  END env {env_i}: funds (engine) = "
                f"P0={int(state.funds[0])} P1={int(state.funds[1])}"
            )
        if env_i >= len(envs):
            break

    out = ROOT / "logs" / f"phase11j_repair_trace_{GID}.txt"
    out.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(LOG)} lines)")


if __name__ == "__main__":
    main()
