#!/usr/bin/env python3
"""
Phase 11J-L2-BUILD-OCCUPIED-SHIP — drill BUILD-OCCUPIED-TILES gids.

Replays each target gid through the oracle pipeline. When a Build action
raises ``UnsupportedOracleAction`` with ``tile occupied``, captures:

- engine state of the build tile and units within radius 2
- PHP unit list at the same envelope frame, restricted to neighborhood
- whether engine's blocker exists in PHP frame (and at what tile)
- active player, day, envelope index, action_stage

Output: ``logs/phase11j_l2_build_occupied_drill.json`` (or ``--out-json``).
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

from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from tools.amarriner_catalog_cos import (  # noqa: E402
    catalog_row_has_both_cos,
    pair_catalog_cos_ids,
)
from tools.desync_audit import CANONICAL_SEED, _seed_for_game  # noqa: E402
from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)


CATALOGS_DEFAULT = [
    ROOT / "data" / "amarriner_gl_std_catalog.json",
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
]
ZIPS_DEFAULT = ROOT / "replays" / "amarriner_gl"
MAP_POOL_DEFAULT = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR_DEFAULT = ROOT / "data" / "maps"


def _load_catalog_union(paths):
    by_id: dict[int, dict[str, Any]] = {}
    for p in paths:
        if not p.is_file():
            continue
        cat = json.loads(p.read_text(encoding="utf-8"))
        games = cat.get("games") or {}
        for g in games.values():
            if isinstance(g, dict) and "games_id" in g:
                gid = int(g["games_id"])
                by_id.setdefault(gid, g)
    return by_id


def _unit_summary(u) -> dict[str, Any]:
    return {
        "type": u.unit_type.name,
        "player": int(u.player),
        "pos": list(u.pos),
        "hp": int(u.hp),
        "ammo": int(u.ammo),
        "fuel": int(u.fuel),
        "unit_id": int(u.unit_id),
        "moved": bool(getattr(u, "moved", False)),
        "captured_to": getattr(u, "capture_to", None),
        "capture_progress": int(getattr(u, "capture_progress", 0)),
        "loaded_ids": [int(c.unit_id) for c in u.loaded_units],
    }


def _engine_neighborhood(state: GameState, r: int, c: int, radius: int = 2) -> list[dict[str, Any]]:
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


def _php_neighborhood(frame: dict[str, Any], r: int, c: int, radius: int = 2) -> list[dict[str, Any]]:
    out = []
    units = (frame.get("units") or {})
    for u in units.values():
        try:
            ur = int(u["y"])
            uc = int(u["x"])
        except (KeyError, TypeError, ValueError):
            continue
        md = abs(ur - r) + abs(uc - c)
        if md > radius:
            continue
        out.append({
            "pos": [ur, uc],
            "md": md,
            "name": u.get("name"),
            "players_id": u.get("players_id"),
            "hit_points": u.get("hit_points"),
            "fuel": u.get("fuel"),
            "ammo": u.get("ammo"),
            "moved": u.get("moved"),
            "carried": u.get("carried"),
            "units_id": u.get("units_id"),
        })
    return out


def _php_unit_at(frame: dict[str, Any], r: int, c: int) -> Optional[dict[str, Any]]:
    units = (frame.get("units") or {})
    for u in units.values():
        try:
            if int(u["y"]) == r and int(u["x"]) == c:
                return u
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _php_find_units_id(frame: dict[str, Any], units_id_target) -> Optional[dict[str, Any]]:
    if units_id_target is None:
        return None
    try:
        target = int(units_id_target)
    except (TypeError, ValueError):
        return None
    units = (frame.get("units") or {})
    for u in units.values():
        try:
            if int(u.get("units_id")) == target:
                return u
        except (TypeError, ValueError):
            continue
    return None


def drill_one(*, games_id: int, zip_path: Path, meta: dict[str, Any],
              map_pool: Path, maps_dir: Path) -> dict[str, Any]:
    random.seed(_seed_for_game(CANONICAL_SEED, games_id))

    capture: dict[str, Any] = {"games_id": games_id, "captured": False}

    envs = parse_p_envelopes_from_zip(zip_path)
    frames = load_replay(zip_path)
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), map_pool, maps_dir)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    engine_to_awbw = {v: k for k, v in awbw_to_engine.items()}
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    actions_applied = 0
    envelopes_applied = 0

    for env_i, (pid, day, actions) in enumerate(envs):
        for j, obj in enumerate(actions):
            kind = obj.get("action") if isinstance(obj, dict) else None

            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    envelope_awbw_player_id=pid,
                )
            except UnsupportedOracleAction as e:
                msg = str(e)
                capture["captured"] = True
                capture["error_msg"] = msg
                capture["env_i"] = env_i
                capture["pid_awbw"] = pid
                capture["day"] = day
                capture["action_idx_in_env"] = j
                capture["action_kind"] = kind
                capture["actions_applied"] = actions_applied
                capture["awbw_to_engine"] = awbw_to_engine
                capture["engine_to_awbw"] = engine_to_awbw

                # Always capture the full Delete action JSON in this envelope
                capture["envelope_delete_actions_full"] = [
                    {"i": i, "obj": a}
                    for i, a in enumerate(actions)
                    if isinstance(a, dict) and a.get("action") == "Delete"
                ]
                # Always capture the full action list of the failing envelope
                capture["envelope_action_summaries"] = [
                    {
                        "i": i,
                        "kind": (a.get("action") if isinstance(a, dict) else None),
                        "player": (a.get("playerID") if isinstance(a, dict) else None),
                        "obj_keys": (list(a.keys())[:20] if isinstance(a, dict) else None),
                        "obj_action_specific": {
                            k: a.get(k) for k in (
                                "action", "playerID", "transportID", "loadedID",
                                "unitId", "newUnit", "unit",
                            ) if isinstance(a, dict) and k in a
                        } if isinstance(a, dict) else None,
                    }
                    for i, a in enumerate(actions)
                ]

                if kind == "Build":
                    gu = obj.get("unit") or obj.get("newUnit") or {}
                    if "global" in gu:
                        gu = gu["global"]
                    try:
                        r = int(gu.get("units_y"))
                        c = int(gu.get("units_x"))
                    except (TypeError, ValueError):
                        r, c = -1, -1
                    capture["build_tile"] = [r, c]
                    capture["build_unit_name"] = gu.get("units_name")
                    capture["build_units_players_id"] = gu.get("units_players_id")
                    eng = awbw_to_engine.get(pid)
                    capture["builder_engine"] = eng
                    capture["active_player"] = int(state.active_player)
                    capture["action_stage"] = str(state.action_stage)
                    capture["funds_before_build"] = {
                        str(p): int(state.funds[p]) for p in (0, 1)
                    }

                    if 0 <= r < state.map_data.height and 0 <= c < state.map_data.width:
                        occ = state.get_unit_at(r, c)
                        capture["engine_unit_at_build_tile"] = (
                            _unit_summary(occ) if occ is not None else None
                        )
                        capture["engine_neighborhood_r2"] = _engine_neighborhood(state, r, c, 2)
                        # Terrain ids of the 4 orth neighbours
                        H = state.map_data.height
                        W = state.map_data.width
                        orth = []
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if not (0 <= nr < H and 0 <= nc < W):
                                continue
                            occ_n = state.get_unit_at(nr, nc)
                            orth.append({
                                "pos": [nr, nc],
                                "tid": int(state.map_data.terrain[nr][nc]),
                                "occupied_by": (
                                    _unit_summary(occ_n) if occ_n is not None else None
                                ),
                            })
                        capture["build_tile_orth_neighbours"] = orth
                        # Reachability of the engine blocker
                        if occ is not None:
                            try:
                                from engine.action import compute_reachable_costs
                                reach = compute_reachable_costs(state, occ)
                                reach_list = []
                                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                    nr, nc = r + dr, c + dc
                                    if not (0 <= nr < H and 0 <= nc < W):
                                        continue
                                    reach_list.append({
                                        "pos": [nr, nc],
                                        "in_reach": (nr, nc) in reach,
                                        "occupied": state.get_unit_at(nr, nc) is not None,
                                    })
                                capture["blocker_reach_orth"] = reach_list
                                capture["blocker_reach_total"] = len(reach)
                            except Exception as e:
                                capture["blocker_reach_error"] = f"{type(e).__name__}: {e}"

                        # PHP frame matched to this envelope. Pre-frame is frame[env_i],
                        # post-frame is frame[env_i + 1].
                        if env_i < len(frames):
                            fb = frames[env_i]
                            php_at = _php_unit_at(fb, r, c)
                            capture["php_unit_at_build_tile_pre_frame"] = php_at
                            capture["php_neighborhood_r2_pre_frame"] = (
                                _php_neighborhood(fb, r, c, 2)
                            )
                            capture["pre_frame_day"] = fb.get("day")
                        if env_i + 1 < len(frames):
                            fa = frames[env_i + 1]
                            php_at = _php_unit_at(fa, r, c)
                            capture["php_unit_at_build_tile_post_frame"] = php_at
                            capture["php_neighborhood_r2_post_frame"] = (
                                _php_neighborhood(fa, r, c, 2)
                            )
                            capture["post_frame_day"] = fa.get("day")

                        # Where did the engine's blocker come from? Look up its
                        # unit_id (engine) in PHP via its units_id when possible.
                        if occ is not None and env_i + 1 < len(frames):
                            # Cross-frame search: did this unit move/die in PHP?
                            # We don't have a cross-mapping engine_unit_id ->
                            # PHP units_id; fall back to PHP units at (occ.pos)
                            # in pre and post frame to compare.
                            opos = occ.pos
                            capture["php_unit_at_engine_blocker_pos_pre"] = (
                                _php_unit_at(frames[env_i], *opos)
                            )
                            capture["php_unit_at_engine_blocker_pos_post"] = (
                                _php_unit_at(frames[env_i + 1], *opos)
                            )

                return capture
            except Exception as e:
                capture["captured"] = True
                capture["python_exception"] = f"{type(e).__name__}: {e}"
                capture["traceback"] = traceback.format_exc(limit=10)
                capture["env_i"] = env_i
                capture["action_idx_in_env"] = j
                capture["action_kind"] = kind
                return capture

            actions_applied += 1
        envelopes_applied += 1

    capture["completed_no_capture"] = True
    capture["actions_applied"] = actions_applied
    capture["envelopes_applied"] = envelopes_applied
    return capture


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, action="append", default=None,
                    help="Catalog JSON; repeatable. Defaults to std + extras.")
    ap.add_argument("--zips-dir", type=Path, default=ZIPS_DEFAULT)
    ap.add_argument("--map-pool", type=Path, default=MAP_POOL_DEFAULT)
    ap.add_argument("--maps-dir", type=Path, default=MAPS_DIR_DEFAULT)
    ap.add_argument("--gid", type=int, action="append", required=True,
                    help="games_id; repeatable")
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / "logs" / "phase11j_l2_build_occupied_drill.json")
    args = ap.parse_args()

    cats = args.catalog if args.catalog else CATALOGS_DEFAULT
    by_id = _load_catalog_union(cats)

    results: list[dict[str, Any]] = []
    for gid in args.gid:
        meta = by_id.get(int(gid))
        zpath = args.zips_dir / f"{gid}.zip"
        if meta is None:
            results.append({"games_id": gid, "error": "missing catalog row"})
            continue
        if not catalog_row_has_both_cos(meta):
            results.append({"games_id": gid, "error": "catalog incomplete cos"})
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
            results.append({
                "games_id": gid,
                "drill_exception": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=20),
            })

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"cases": results}, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(args.out_json), "n": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
