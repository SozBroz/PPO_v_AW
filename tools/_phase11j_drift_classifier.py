#!/usr/bin/env python3
"""Phase 11J-BUILD-NO-OP-CLUSTER-CLOSE — classify drift sources across 12 gids.

For each gid, walks the per-envelope drift trace; for each NEW funds drift
event (delta changing between envelopes), it instruments the repair phase to
record which units were healed and what the engine vs PHP cost should have
been.

Heuristic for "PHP-canon" repair cost:
  display_hp = ceil(internal_hp / 10)
  if display_hp >= 10                 → 0 (PHP refuses)
  elif display_hp == 9 and not Rachel → 10% of unit cost (+1 display)
  elif display_hp == 9 and Rachel     → 20% of unit cost (+2 display)
  else (display_hp <= 8) and not R    → 20% of unit cost (+2 display)
  else                                → 30% of unit cost (Rachel +3)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state, _property_day_repair_gold
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

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
CATALOG_EXTRAS = ROOT / "data" / "amarriner_gl_extras_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _php_canon_cost(unit_type, internal_hp: int, is_rachel: bool) -> int:
    listed = UNIT_STATS[unit_type].cost
    if listed <= 0:
        return 0
    display = math.ceil(internal_hp / 10) if internal_hp > 0 else 0
    if display >= 10:
        return 0
    if is_rachel:
        if display == 9:
            return max(1, (20 * listed) // 100)  # +2 display, capped
        return max(1, (30 * listed) // 100)      # +3 display
    if display == 9:
        return max(1, (10 * listed) // 100)
    return max(1, (20 * listed) // 100)


def trace_one(gid: int) -> dict[str, Any]:
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

    repair_log: list[dict] = []
    cur_env = [None]
    cur_day = [None]

    orig = GameState._resupply_on_properties

    def patched(self, player):
        co = self.co_states[player]
        is_rachel = co.co_id == 28
        property_heal = 30 if is_rachel else 20
        eligible = []
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
            if not (qualifies and not prop.is_lab and not prop.is_comm_tower
                    and unit.hp < 100):
                continue
            step = min(property_heal, 100 - unit.hp)
            eng_cost = _property_day_repair_gold(step, unit.unit_type)
            php_cost = _php_canon_cost(unit.unit_type, unit.hp, is_rachel)
            eligible.append({
                "type": unit.unit_type.name,
                "hp": unit.hp,
                "display_hp": math.ceil(unit.hp / 10) if unit.hp > 0 else 0,
                "engine_cost": eng_cost,
                "php_canon_cost": php_cost,
                "delta_per_unit": eng_cost - php_cost,
            })
        before = int(self.funds[player])
        orig(self, player)
        after = int(self.funds[player])
        if eligible:
            repair_log.append({
                "env": cur_env[0],
                "day": cur_day[0],
                "player": player,
                "co": "Rachel" if is_rachel else "other",
                "engine_spent": before - after,
                "php_canon_total": sum(u["php_canon_cost"] for u in eligible),
                "engine_minus_php_step": (before - after) - sum(
                    u["php_canon_cost"] for u in eligible),
                "units": eligible,
            })

    GameState._resupply_on_properties = patched

    drift_events = []
    last_delta = {0: 0, 1: 0}
    try:
        for env_i, (pid, day, actions) in enumerate(envs):
            cur_env[0] = env_i
            cur_day[0] = day
            for obj in actions:
                apply_oracle_action_json(state, obj, awbw_to_engine,
                                         envelope_awbw_player_id=pid)
            frame_after = frames[env_i + 1] if env_i + 1 < len(frames) else None
            if frame_after is None:
                continue
            php_funds = {0: 0, 1: 0}
            for k, pl in (frame_after.get("players") or {}).items():
                try:
                    apid = int(pl.get("id"))
                    if apid in awbw_to_engine:
                        php_funds[awbw_to_engine[apid]] = int(pl.get("funds") or 0)
                except (TypeError, ValueError):
                    pass
            cur = {p: int(state.funds[p]) - php_funds[p] for p in (0, 1)}
            for p in (0, 1):
                if cur[p] != last_delta[p]:
                    drift_events.append({
                        "env": env_i, "day": day, "pid": pid,
                        "player": p,
                        "delta_was": last_delta[p],
                        "delta_now": cur[p],
                        "delta_change": cur[p] - last_delta[p],
                    })
            last_delta = cur
    except Exception as e:
        pass
    finally:
        GameState._resupply_on_properties = orig

    return {
        "gid": gid,
        "co_p0": co0, "co_p1": co1,
        "drift_events": drift_events,
        "repair_log": repair_log,
        "final_drift": last_delta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, action="append", required=True)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "logs" / "phase11j_drift_classify.json")
    args = ap.parse_args()
    cases = [trace_one(g) for g in args.gid]
    args.out.write_text(
        json.dumps({"cases": cases}, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")

    for c in cases:
        gid = c["gid"]
        repair_drift_total = sum(r["engine_minus_php_step"] for r in c["repair_log"])
        per_player = {0: 0, 1: 0}
        for r in c["repair_log"]:
            per_player[r["player"]] += r["engine_minus_php_step"]
        print(f"\ngid={gid}  repair_drift_total={repair_drift_total}  per_player={per_player}  final_drift={c['final_drift']}")
        # Show top 5 phantom-repair events
        with_drift = [r for r in c["repair_log"] if r["engine_minus_php_step"] != 0]
        for r in with_drift[:8]:
            culprits = [f"{u['type']}@hp{u['hp']}(d{u['display_hp']}):eng{u['engine_cost']}/php{u['php_canon_cost']}"
                        for u in r["units"] if u["delta_per_unit"] != 0]
            print(f"  env={r['env']} day={r['day']} P{r['player']} step_drift={r['engine_minus_php_step']}  {culprits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
