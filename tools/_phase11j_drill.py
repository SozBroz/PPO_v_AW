#!/usr/bin/env python3
"""
Phase 11J — Fire-drift drilldown.

Replays each target zip through the oracle pipeline and intercepts the
`_apply_attack` call to capture engine state at the moment of failure:
- attacker resolved at unit_pos (type, hp, ammo, fuel, owner)
- defender at target_pos (type, hp, owner) and full neighborhood
- get_attack_targets(state, attacker, move_pos) and unit.pos
- nearby friendly/enemy units within attack range of move_pos
- the oracle Fire envelope (move paths + combatInfo) leading to it
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import Action, ActionType, get_attack_targets  # noqa: E402
from engine.combat import get_base_damage  # noqa: E402
from engine.game import GameState  # noqa: E402
from engine.unit import UNIT_STATS  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from tools.amarriner_catalog_cos import (  # noqa: E402
    catalog_row_has_both_cos,
    pair_catalog_cos_ids,
)
from tools.desync_audit import CANONICAL_SEED, _seed_for_game  # noqa: E402
from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)

CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS_DEFAULT = ROOT / "replays" / "amarriner_gl"
MAP_POOL_DEFAULT = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR_DEFAULT = ROOT / "data" / "maps"


def _meta_int(meta: dict[str, Any], key: str, default: int = -1) -> int:
    v = meta.get(key, default)
    if v is None:
        return default
    return int(v)


def _unit_summary(u) -> dict[str, Any]:
    return {
        "unit_type": u.unit_type.name,
        "player": int(u.player),
        "pos": list(u.pos),
        "hp": int(u.hp),
        "ammo": int(u.ammo),
        "fuel": int(u.fuel),
        "unit_id": int(u.unit_id),
        "is_alive": bool(u.is_alive),
        "loaded_ids": [int(c.unit_id) for c in u.loaded_units],
    }


def _all_units_at(state: GameState, r: int, c: int) -> list[dict[str, Any]]:
    out = []
    for p in (0, 1):
        for u in state.units[p]:
            if u.is_alive and u.pos == (r, c):
                out.append(_unit_summary(u))
    return out


def _neighborhood(state: GameState, r: int, c: int, radius: int = 3) -> list[dict[str, Any]]:
    out = []
    H = state.map_data.height
    W = state.map_data.width
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            tr, tc = r + dr, c + dc
            if not (0 <= tr < H and 0 <= tc < W):
                continue
            occ = state.get_unit_at(tr, tc)
            if occ is not None:
                out.append({
                    "pos": [tr, tc],
                    "md": abs(dr) + abs(dc),
                    "tid": int(state.map_data.terrain[tr][tc]),
                    "unit": _unit_summary(occ),
                })
    return out


def drill_one(
    *,
    games_id: int,
    zip_path: Path,
    meta: dict[str, Any],
    map_pool: Path,
    maps_dir: Path,
) -> dict[str, Any]:
    from engine.game import make_initial_state

    random.seed(_seed_for_game(CANONICAL_SEED, games_id))

    capture: dict[str, Any] = {"games_id": games_id, "captured": False}

    orig_apply = GameState._apply_attack

    def _instrumented(self: GameState, action: Action):
        try:
            return orig_apply(self, action)
        except ValueError as e:
            if capture.get("captured"):
                raise
            msg = str(e)
            capture["captured"] = True
            capture["error_msg"] = msg
            attacker = self.get_unit_at(*action.unit_pos) if action.unit_pos else None
            defender = (
                self.get_unit_at(*action.target_pos)
                if action.target_pos else None
            )
            atk_from = action.move_pos if action.move_pos is not None else (
                attacker.pos if attacker else None
            )

            cap = {
                "action": {
                    "type": action.action_type.name,
                    "unit_pos": list(action.unit_pos) if action.unit_pos else None,
                    "move_pos": list(action.move_pos) if action.move_pos else None,
                    "target_pos": list(action.target_pos) if action.target_pos else None,
                    "select_unit_id": getattr(action, "select_unit_id", None),
                },
                "active_player": int(self.active_player),
                "action_stage": str(self.action_stage),
                "selected_unit": (
                    _unit_summary(self.selected_unit)
                    if self.selected_unit is not None else None
                ),
                "selected_move_pos": (
                    list(self.selected_move_pos)
                    if self.selected_move_pos is not None else None
                ),
                "attacker_at_unit_pos": _unit_summary(attacker) if attacker else None,
                "defender_at_target_pos": _unit_summary(defender) if defender else None,
                "atk_from": list(atk_from) if atk_from else None,
                "all_units_at_unit_pos": (
                    _all_units_at(self, *action.unit_pos) if action.unit_pos else []
                ),
                "all_units_at_target_pos": (
                    _all_units_at(self, *action.target_pos) if action.target_pos else []
                ),
                "all_units_at_move_pos": (
                    _all_units_at(self, *action.move_pos) if action.move_pos else []
                ),
            }
            if attacker is not None and atk_from is not None:
                tgts = get_attack_targets(self, attacker, tuple(atk_from))
                cap["attack_targets_from_atk_from"] = [list(t) for t in tgts]
                cap["attack_targets_from_unit_pos"] = [
                    list(t) for t in get_attack_targets(self, attacker, attacker.pos)
                ]
                stats = UNIT_STATS[attacker.unit_type]
                cap["attacker_stats"] = {
                    "min_range": stats.min_range,
                    "max_range": stats.max_range,
                    "is_indirect": bool(stats.is_indirect),
                    "max_ammo": stats.max_ammo,
                }
                if defender is not None:
                    cap["base_damage_atk_vs_def"] = get_base_damage(
                        attacker.unit_type, defender.unit_type
                    )
            if attacker is not None:
                cap["neighborhood_around_attacker"] = _neighborhood(
                    self, *attacker.pos, radius=4
                )
            if action.target_pos:
                cap["neighborhood_around_target"] = _neighborhood(
                    self, *action.target_pos, radius=3
                )
            capture.update(cap)
            raise

    GameState._apply_attack = _instrumented  # type: ignore[assignment]

    try:
        envs = parse_p_envelopes_from_zip(zip_path)
        frames = load_replay(zip_path)
        co0, co1 = pair_catalog_cos_ids(meta)
        mid = _meta_int(meta, "map_id")
        map_data = load_map(mid, map_pool, maps_dir)
        awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
        first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
        state = make_initial_state(
            map_data, co0, co1, starting_funds=0,
            tier_name=str(meta.get("tier") or "T2"),
            replay_first_mover=first_mover,
        )

        actions_applied = 0
        envelopes_applied = 0
        last_envelope: Optional[tuple[int, int, list[dict[str, Any]]]] = None
        last_action_obj: Optional[dict[str, Any]] = None

        for env_i, (pid, day, actions) in enumerate(envs):
            for j, obj in enumerate(actions):
                last_envelope = (pid, day, actions)
                last_action_obj = obj
                try:
                    apply_oracle_action_json(
                        state, obj, awbw_to_engine,
                        envelope_awbw_player_id=pid,
                    )
                except UnsupportedOracleAction as e:
                    capture.setdefault("oracle_unsupported", str(e))
                    capture["last_envelope_idx"] = env_i
                    capture["last_action_idx_in_env"] = j
                    capture["actions_applied"] = actions_applied
                    capture["last_action_kind"] = obj.get("action")
                    capture["last_envelope_pid"] = pid
                    capture["last_envelope_day"] = day
                    capture["last_envelope_actions"] = actions
                    capture["last_action_obj"] = obj
                    return capture
                except Exception as e:
                    capture.setdefault("python_exception", f"{type(e).__name__}: {e}")
                    capture.setdefault("traceback", traceback.format_exc(limit=10))
                    capture["last_envelope_idx"] = env_i
                    capture["last_action_idx_in_env"] = j
                    capture["actions_applied"] = actions_applied
                    capture["last_action_kind"] = obj.get("action")
                    capture["last_envelope_pid"] = pid
                    capture["last_envelope_day"] = day
                    capture["last_envelope_actions"] = actions
                    capture["last_action_obj"] = obj
                    return capture
                actions_applied += 1
            envelopes_applied += 1
        capture["completed"] = True
        capture["actions_applied"] = actions_applied
        capture["envelopes_applied"] = envelopes_applied
        return capture
    finally:
        GameState._apply_attack = orig_apply  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    ap.add_argument("--zips-dir", type=Path, default=ZIPS_DEFAULT)
    ap.add_argument("--map-pool", type=Path, default=MAP_POOL_DEFAULT)
    ap.add_argument("--maps-dir", type=Path, default=MAPS_DIR_DEFAULT)
    ap.add_argument("--games-id", type=int, action="append", required=True)
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "logs" / "phase11j_drill.json")
    args = ap.parse_args()

    cat = json.loads(args.catalog.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id = {int(g["games_id"]): g for g in games.values()
             if isinstance(g, dict) and "games_id" in g}

    std_maps = gl_std_map_ids(args.map_pool)
    results: list[dict[str, Any]] = []

    for gid in args.games_id:
        meta = by_id.get(int(gid))
        zpath = args.zips_dir / f"{gid}.zip"
        if meta is None:
            results.append({"games_id": gid, "error": "missing catalog row"})
            continue
        if not catalog_row_has_both_cos(meta):
            results.append({"games_id": gid, "error": "catalog incomplete cos"})
            continue
        mid = _meta_int(meta, "map_id")
        if mid not in std_maps:
            results.append({"games_id": gid, "error": f"map_id {mid} not in std pool"})
            continue
        if not zpath.is_file():
            results.append({"games_id": gid, "error": f"missing zip {zpath}"})
            continue
        try:
            results.append(drill_one(
                games_id=int(gid), zip_path=zpath, meta=meta,
                map_pool=args.map_pool, maps_dir=args.maps_dir,
            ))
        except Exception as e:
            results.append({"games_id": gid, "drill_exception": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(limit=15)})

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"cases": results}, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(args.out_json), "n": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
