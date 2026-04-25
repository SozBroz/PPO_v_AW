"""Phase 11J-VONBOLT-SCOP-SHIP — drill the 3 Von Bolt SCOP gids.

For each ``games_id`` in ``GIDS`` we:
  * Walk the PHP envelope stream and locate the Von Bolt SCOP (coName="Von Bolt",
    coPower="S") envelope.
  * Print ``missileCoords`` (chosen center) + the AWBW-canon 5x5 diamond AOE
    (Manhattan radius 2) AND the engine's current 3x3 box AOE. These differ:
    diamond=13 squares, box=9 squares.
  * Step the engine up to and through the SCOP envelope. Dump engine unit HP
    and ``moved`` flag for opponent units in the diamond AOE — *plus* compare
    against the PHP snapshot frame at the same envelope index.
  * Walk the next opponent envelope and report which (if any) of the affected
    enemy units PHP attempts to move/attack/wait — they should all be stunned
    (no actions emitted from those positions in the next opponent turn).

Output is appended to logs/_phase11j_vonbolt_scop_drill.json for the report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    apply_oracle_action_json,
    parse_p_envelopes_from_zip,
)


GIDS = (1621434, 1621898, 1622328)
ZIPS_DIR = Path("replays/amarriner_gl")
MAP_POOL = Path("data/gl_map_pool.json")
MAPS_DIR = Path("data/maps")
OUT = Path("logs/_phase11j_vonbolt_scop_drill.json")


def _aoe_box(cy: int, cx: int) -> set[tuple[int, int]]:
    return {(cy + dr, cx + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _aoe_diamond(cy: int, cx: int, r: int = 2) -> set[tuple[int, int]]:
    return {
        (cy + dr, cx + dc)
        for dr in range(-r, r + 1)
        for dc in range(-r, r + 1)
        if abs(dr) + abs(dc) <= r
    }


def _build_state(zip_path: Path) -> tuple[GameState, dict[int, int], list[dict]]:
    frames = load_replay(zip_path)
    if not frames:
        raise RuntimeError(f"empty replay: {zip_path}")
    f0 = frames[0]
    map_id = int(f0["maps_id"])
    map_data = load_map(map_id, MAP_POOL, MAPS_DIR)

    awbw_to_engine: dict[int, int] = {}
    p_blocks_raw = f0.get("players") or {}
    if isinstance(p_blocks_raw, dict):
        p_blocks = [p_blocks_raw[k] for k in sorted(p_blocks_raw)]
    else:
        p_blocks = list(p_blocks_raw)
    for idx, pb in enumerate(p_blocks):
        awbw_to_engine[int(pb["id"])] = idx

    co_ids = [int(pb.get("co_id") or 1) for pb in p_blocks]
    state = make_initial_state(
        map_data=map_data,
        p0_co_id=co_ids[0],
        p1_co_id=co_ids[1] if len(co_ids) > 1 else co_ids[0],
        starting_funds=int(f0.get("starting_funds") or 0),
        tier_name="probe",
    )
    return state, awbw_to_engine, frames


def _find_vonbolt_scop_env(envelopes: list[tuple[int, int, list[dict]]]) -> int | None:
    for i, (_pid, _day, acts) in enumerate(envelopes):
        for obj in acts:
            if (
                obj.get("action") == "Power"
                and str(obj.get("coName") or "") == "Von Bolt"
                and str(obj.get("coPower") or "").upper() == "S"
            ):
                return i
    return None


def _missile_center(envelope_actions: list[dict]) -> tuple[int, int] | None:
    for obj in envelope_actions:
        if obj.get("action") == "Power" and str(obj.get("coName") or "") == "Von Bolt":
            mc = obj.get("missileCoords")
            if isinstance(mc, list) and mc:
                first = mc[0]
                if isinstance(first, dict):
                    try:
                        return int(first["y"]), int(first["x"])
                    except Exception:
                        return None
    return None


def _php_units_in_set(php_frame: dict, positions: set[tuple[int, int]]) -> list[dict]:
    out = []
    units_blob = php_frame.get("units") or {}
    if isinstance(units_blob, dict):
        iter_src = units_blob.values()
    else:
        iter_src = units_blob
    for u in iter_src:
        if not isinstance(u, dict):
            continue
        try:
            y = int(u["units_y"])
            x = int(u["units_x"])
        except (KeyError, ValueError, TypeError):
            continue
        if (y, x) not in positions:
            continue
        out.append({
            "row": y,
            "col": x,
            "player": int(u.get("players_id") or -1),
            "name": str(u.get("units_name") or "?"),
            "hp": float(u.get("units_hit_points") or 0),
        })
    return out


def _engine_units_in_set(state: GameState, positions: set[tuple[int, int]]) -> list[dict]:
    out = []
    for player in (0, 1):
        for u in state.units[player]:
            if not u.is_alive:
                continue
            if u.pos not in positions:
                continue
            out.append({
                "row": u.pos[0],
                "col": u.pos[1],
                "player": int(u.player),
                "type": u.unit_type.name,
                "hp_internal": int(u.hp),
                "moved": bool(u.moved),
                "is_stunned": bool(getattr(u, "is_stunned", False)),
            })
    return out


def drill_one(gid: int) -> dict:
    zip_path = ZIPS_DIR / f"{gid}.zip"
    state, awbw_to_engine, frames = _build_state(zip_path)
    envelopes = parse_p_envelopes_from_zip(zip_path)
    scop_idx = _find_vonbolt_scop_env(envelopes)
    if scop_idx is None:
        return {"gid": gid, "error": "no_vonbolt_scop_envelope"}

    center = _missile_center(envelopes[scop_idx][2])
    if center is None:
        return {"gid": gid, "error": "no_missile_coords"}

    cy, cx = center
    box = _aoe_box(cy, cx)
    diamond = _aoe_diamond(cy, cx, r=2)

    # Step engine through and including the SCOP envelope.
    for env_i, (pid, _day, acts) in enumerate(envelopes[: scop_idx + 1]):
        for obj in acts:
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine, envelope_awbw_player_id=pid
                )
            except Exception as exc:
                return {
                    "gid": gid,
                    "scop_envelope": scop_idx,
                    "error": f"replay_failed at env {env_i}: {type(exc).__name__}: {exc}",
                }

    # Capture state immediately after SCOP fires.
    engine_diamond = _engine_units_in_set(state, diamond)
    engine_box = _engine_units_in_set(state, box)
    engine_corners_only = [u for u in engine_diamond if (u["row"], u["col"]) not in box]

    php_post_scop = frames[scop_idx + 1] if scop_idx + 1 < len(frames) else None
    php_diamond = _php_units_in_set(php_post_scop, diamond) if php_post_scop else []

    # Walk the next opponent envelope: did PHP try to act on any stunned unit?
    vb_player = state.active_player  # opponent of Von Bolt becomes active after SCOP+End
    next_opp_actions: list[dict] = []
    for env_i in range(scop_idx + 1, len(envelopes)):
        pid, day, acts = envelopes[env_i]
        eng_p = awbw_to_engine.get(pid)
        if eng_p is None:
            continue
        if eng_p == vb_player:
            # Still Von Bolt's turn (e.g., end-turn envelope follows). Skip.
            for obj in acts:
                if obj.get("action") == "End":
                    break
            continue
        # First opponent envelope.
        for obj in acts:
            kind = obj.get("action")
            move = obj.get("Move") or obj
            paths = move.get("paths") if isinstance(move, dict) else None
            origin = None
            if isinstance(paths, list) and paths:
                first_step = paths[0]
                if isinstance(first_step, dict):
                    try:
                        origin = (int(first_step["y"]), int(first_step["x"]))
                    except Exception:
                        pass
            if origin is None:
                continue
            stunned = origin in diamond
            next_opp_actions.append({
                "kind": kind,
                "origin": origin,
                "stunned_diamond": stunned,
            })
        break

    illegal_attempts = [a for a in next_opp_actions if a["stunned_diamond"]]

    return {
        "gid": gid,
        "scop_envelope": scop_idx,
        "envelopes_total": len(envelopes),
        "scop_center_yx": [cy, cx],
        "engine_3x3_box_size": len(box),
        "awbw_5x5_diamond_size": len(diamond),
        "engine_units_in_3x3_box_post_scop": engine_box,
        "engine_units_in_diamond_outside_box": engine_corners_only,
        "php_units_in_diamond_post_scop": php_diamond,
        "next_opponent_envelope_actions": next_opp_actions,
        "next_opponent_actions_originating_inside_diamond": illegal_attempts,
        "diamond_minus_box_size": len(diamond - box),
    }


def main() -> None:
    out: list[dict] = []
    for gid in GIDS:
        try:
            r = drill_one(gid)
        except Exception as exc:  # noqa: BLE001
            r = {"gid": gid, "fatal": f"{type(exc).__name__}: {exc}"}
        out.append(r)
        print(json.dumps(r, indent=2))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[drill] wrote {OUT}")


if __name__ == "__main__":
    main()
