#!/usr/bin/env python3
"""
Phase 11K — 200-game silent drift drill (funds / HP≥10 / unit-count first indices).

READ-ONLY harness: imports engine + replay_state_diff helpers; does not edit
production modules. Writes ``logs/phase11k_sample_gids.txt`` and
``logs/phase11k_drift_data.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from engine.unit import UNIT_STATS  # noqa: E402
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
from tools.replay_snapshot_compare import (  # noqa: E402
    compare_funds,
    replay_snapshot_pairing,
)

REGISTER_DEFAULT = ROOT / "logs" / "desync_register_post_phase10q.jsonl"
CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS_DEFAULT = ROOT / "replays" / "amarriner_gl"
MAP_POOL_DEFAULT = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR_DEFAULT = ROOT / "data" / "maps"
SAMPLE_OUT = ROOT / "logs" / "phase11k_sample_gids.txt"
DRIFT_OUT = ROOT / "logs" / "phase11k_drift_data.jsonl"

HP_INTERNAL_THRESHOLD = 10


def _meta_int(meta: dict[str, Any], key: str, default: int = -1) -> int:
    v = meta.get(key, default)
    if v is None:
        return default
    return int(v)


def load_catalog(path: Path) -> dict[int, dict[str, Any]]:
    cat = json.loads(path.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id: dict[int, dict[str, Any]] = {}
    for _k, g in games.items():
        if isinstance(g, dict) and "games_id" in g:
            by_id[int(g["games_id"])] = g
    return by_id


def load_ok_rows(register_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with register_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("class") == "ok":
                rows.append(r)
    return rows


def env_length_bucket(n_env: int, lo: int, hi: int) -> str:
    if n_env <= lo:
        return "short"
    if n_env <= hi:
        return "mid"
    return "long"


def stratified_sample_gids(
    rows: list[dict[str, Any]],
    n_sample: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """
    Stratify by tier (T0–T4) and envelope count tertiles (short/mid/long).
    Deterministic largest-remainder allocation to hit exactly ``n_sample``.
    """
    env_vals = sorted(int(r.get("envelopes_total") or 0) for r in rows)
    if not env_vals:
        return [], {"error": "no rows"}
    n = len(env_vals)
    lo = env_vals[max(0, n // 3 - 1)]
    hi = env_vals[max(0, (2 * n) // 3 - 1)]

    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        gid = int(r["games_id"])
        tier = str(r.get("tier") or "T?")
        if tier not in {f"T{i}" for i in range(5)}:
            tier = "T?"
        ne = int(r.get("envelopes_total") or 0)
        cells[(tier, env_length_bucket(ne, lo, hi))].append(gid)

    for k in cells:
        cells[k].sort()

    rng = random.Random(seed)
    total_pop = len(rows)
    # targets per cell (float)
    targets: dict[tuple[str, str], float] = {}
    for key, gids in cells.items():
        targets[key] = n_sample * len(gids) / total_pop

    floors: dict[tuple[str, str], int] = {k: int(math.floor(targets[k])) for k in targets}
    rem = n_sample - sum(floors.values())
    # distribute remainder to cells with largest fractional part
    frac = sorted(
        ((targets[k] - floors[k], k) for k in targets),
        key=lambda t: (-t[0], str(t[1])),
    )
    picks: dict[tuple[str, str], int] = dict(floors)
    for i in range(rem):
        picks[frac[i][1]] += 1

    out: list[int] = []
    meta_cells: dict[str, Any] = {
        "seed": seed,
        "n_ok_register": total_pop,
        "envelope_tertile_cutoffs": {"lo": lo, "hi": hi},
        "cells": {},
    }

    for key, gids in sorted(cells.items()):
        need = picks.get(key, 0)
        if need <= 0:
            meta_cells["cells"][f"{key[0]}_{key[1]}"] = {
                "population": len(gids),
                "sampled": 0,
            }
            continue
        need = min(need, len(gids))
        # deterministic shuffle per cell
        idx = list(range(len(gids)))
        rng.shuffle(idx)
        chosen = sorted(gids[i] for i in idx[:need])
        out.extend(chosen)
        meta_cells["cells"][f"{key[0]}_{key[1]}"] = {
            "population": len(gids),
            "sampled": len(chosen),
        }

    # If short (impossible strata), top up from global pool
    if len(out) < n_sample:
        pool = sorted({int(r["games_id"]) for r in rows} - set(out))
        rng.shuffle(pool)
        for gid in pool:
            if len(out) >= n_sample:
                break
            out.append(gid)
        out.sort()
        meta_cells["top_up"] = n_sample - sum(
            meta_cells["cells"][k].get("sampled", 0)
            for k in meta_cells["cells"]
        )

    out = sorted(out)[:n_sample]
    meta_cells["final_sample_size"] = len(out)
    return out, meta_cells


def _php_internal_hp(pu: dict[str, Any]) -> Optional[int]:
    hp = pu.get("hit_points")
    if hp is None:
        return None
    return int(round(float(hp) * 10))


def _build_tile_maps(
    php_frame: dict[str, Any], state: Any, awbw_to_engine: dict[int, int]
) -> tuple[dict[tuple[int, int, int], dict[str, Any]], dict[tuple[int, int, int], Any]]:
    php_by_tile: dict[tuple[int, int, int], dict[str, Any]] = {}
    for _k, u in (php_frame.get("units") or {}).items():
        if not isinstance(u, dict):
            continue
        if str(u.get("carried", "N")).upper() == "Y":
            continue
        col, row = int(u["x"]), int(u["y"])
        pid = int(u["players_id"])
        eng_seat = awbw_to_engine[pid]
        key = (eng_seat, row, col)
        php_by_tile[key] = u

    eng_by_tile: dict[tuple[int, int, int], Any] = {}
    for seat in (0, 1):
        for u in state.units[seat]:
            if u.is_alive:
                r, c = u.pos
                eng_by_tile[(seat, r, c)] = u
    return php_by_tile, eng_by_tile


def _type_mismatch(pu: dict[str, Any], eu: Any) -> bool:
    php_name = str(pu.get("name", "")).strip()
    eng_name = UNIT_STATS[eu.unit_type].name
    if not php_name:
        return False
    aliases = {
        "Md.Tank": "Medium Tank",
        "Md. Tank": "Medium Tank",
    }
    eng_cmp = aliases.get(eng_name, eng_name)
    php_cmp = aliases.get(php_name, php_name)
    return eng_cmp != php_cmp


def _last_action_kind(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "EMPTY"
    a = str(actions[-1].get("action") or "?")
    return a


def _drill_one_clean(games_id: int, meta: dict[str, Any], zips_dir: Path, map_pool: Path, maps_dir: Path) -> dict[str, Any]:
    """Single-game drill with deduped HP≥10 logic."""
    random.seed(_seed_for_game(CANONICAL_SEED, games_id))
    zip_path = zips_dir / f"{games_id}.zip"
    mid = _meta_int(meta, "map_id")
    co0, co1 = pair_catalog_cos_ids(meta)
    tier_name = str(meta.get("tier") or "T2")

    row: dict[str, Any] = {
        "games_id": games_id,
        "map_id": mid,
        "tier": tier_name,
        "co_p0_id": co0,
        "co_p1_id": co1,
        "n_frames": None,
        "n_envelopes": None,
        "pairing": None,
        "oracle_error": None,
        "replay_truncated": False,
        "replay_truncated_reason": None,
        "initial_mismatch": False,
        "first_funds_step": None,
        "first_hp10_step": None,
        "first_count_step": None,
        "first_any_step": None,
        "clean_snapshot_through_stop": None,
        "silent_drift": None,
        "events": {},
    }

    if not zip_path.is_file():
        row["oracle_error"] = "missing zip"
        return row

    try:
        frames = load_replay(zip_path)
    except Exception as e:
        row["oracle_error"] = f"load_replay: {type(e).__name__}: {e}"
        return row

    envs = parse_p_envelopes_from_zip(zip_path)
    row["n_frames"] = len(frames)
    row["n_envelopes"] = len(envs)

    if not envs:
        row["oracle_error"] = "no p: envelopes"
        return row

    pairing = replay_snapshot_pairing(len(frames), len(envs))
    row["pairing"] = pairing
    if pairing is None:
        row["oracle_error"] = (
            f"unsupported snapshot layout: {len(frames)} frames vs {len(envs)} envs"
        )
        return row

    try:
        awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    except Exception as e:
        row["oracle_error"] = f"map_snapshot_player_ids_to_engine: {e}"
        return row

    map_data = load_map(mid, map_pool, maps_dir)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data,
        co0,
        co1,
        starting_funds=0,
        tier_name=tier_name,
        replay_first_mover=first_mover,
    )

    from tools.replay_snapshot_compare import compare_snapshot_to_engine  # noqa: PLC0415

    init_mm = compare_snapshot_to_engine(frames[0], state, awbw_to_engine)
    if init_mm:
        row["initial_mismatch"] = True
        row["oracle_error"] = "frame0 mismatch: " + "; ".join(init_mm[:4])
        return row

    first_funds: Optional[int] = None
    first_hp10: Optional[int] = None
    first_count: Optional[int] = None
    first_full_snapshot: Optional[int] = None  # same ordering as replay_state_diff / 10F

    def record_event(
        kind: str,
        step_i: int,
        pid: int,
        day: int,
        actions: list[dict[str, Any]],
    ) -> None:
        co_active = int(state.co_states[int(state.active_player)].co_id)
        row["events"][kind] = {
            "step_i": step_i,
            "envelope_awbw_player_id": int(pid),
            "envelope_day_field": int(day),
            "engine_turn": int(state.turn),
            "engine_active_player": int(state.active_player),
            "active_co_id": co_active,
            "last_action_kind": _last_action_kind(actions),
        }

    for step_i, (pid, day, actions) in enumerate(envs):
        for obj in actions:
            try:
                apply_oracle_action_json(
                    state,
                    obj,
                    awbw_to_engine,
                    envelope_awbw_player_id=pid,
                )
            except UnsupportedOracleAction as e:
                row["oracle_error"] = f"step {step_i} UnsupportedOracleAction: {e}"
                break
            except Exception as e:
                row["oracle_error"] = f"step {step_i} {type(e).__name__}: {e}"
                break
            if state.done:
                row["replay_truncated"] = True
                row["replay_truncated_reason"] = (
                    "Game ended before zip exhausted (e.g. Resign/Victory)"
                )
                break
        if row.get("oracle_error"):
            break

        snap_i = step_i + 1
        if snap_i >= len(frames):
            continue

        php_f = frames[snap_i]
        full_mm = compare_snapshot_to_engine(php_f, state, awbw_to_engine)
        if first_full_snapshot is None and full_mm:
            first_full_snapshot = step_i
            row["first_mismatch_lines"] = full_mm[:16]
            if "first_full_snapshot" not in row["events"]:
                record_event("first_full_snapshot", step_i, pid, day, actions)

        funds_mm = compare_funds(php_f, state, awbw_to_engine)
        if first_funds is None and funds_mm:
            first_funds = step_i
            record_event("first_funds", step_i, pid, day, actions)

        php_by_tile, eng_by_tile = _build_tile_maps(php_f, state, awbw_to_engine)
        if first_count is None:
            if len(php_by_tile) != len(eng_by_tile) or set(php_by_tile) != set(
                eng_by_tile
            ):
                first_count = step_i
                record_event("first_count", step_i, pid, day, actions)

        if first_hp10 is None:
            for key in set(php_by_tile) & set(eng_by_tile):
                pu, eu = php_by_tile[key], eng_by_tile[key]
                if _type_mismatch(pu, eu):
                    continue
                php_i = _php_internal_hp(pu)
                if php_i is None:
                    continue
                if abs(int(eu.hp) - php_i) >= HP_INTERNAL_THRESHOLD:
                    first_hp10 = step_i
                    record_event("first_hp10", step_i, pid, day, actions)
                    break

    row["first_funds_step"] = first_funds
    row["first_hp10_step"] = first_hp10
    row["first_count_step"] = first_count
    row["first_snapshot_mismatch_step"] = first_full_snapshot
    candidates = [x for x in (first_funds, first_hp10, first_count) if x is not None]
    row["first_any_step"] = min(candidates) if candidates else None
    row["clean_snapshot_through_stop"] = first_full_snapshot is None
    # Phase 10F parity: drift when combined snapshot compare fails (funds before units).
    row["silent_drift"] = first_full_snapshot is not None
    row["oracle_aborted"] = bool(row.get("oracle_error"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register", type=Path, default=REGISTER_DEFAULT)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    ap.add_argument("--zips-dir", type=Path, default=ZIPS_DEFAULT)
    ap.add_argument("--map-pool", type=Path, default=MAP_POOL_DEFAULT)
    ap.add_argument("--maps-dir", type=Path, default=MAPS_DIR_DEFAULT)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-out", type=Path, default=SAMPLE_OUT)
    ap.add_argument("--drift-out", type=Path, default=DRIFT_OUT)
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--games-id", type=int, action="append", default=None)
    args = ap.parse_args()

    rows = load_ok_rows(args.register)
    by_cat = load_catalog(args.catalog)
    std_maps = gl_std_map_ids(args.map_pool)

    gids: list[int]
    strat_meta: dict[str, Any]
    if args.games_id:
        gids = sorted(set(args.games_id))
        strat_meta = {"mode": "explicit", "gids": gids}
    else:
        gids, strat_meta = stratified_sample_gids(rows, args.sample_size, args.seed)

    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    args.sample_out.write_text(
        "\n".join(str(g) for g in gids) + "\n",
        encoding="utf-8",
    )

    summary_path = args.sample_out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(strat_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"wrote_sample": str(args.sample_out), "n": len(gids)}, indent=2))

    if args.sample_only:
        return 0

    args.drift_out.parent.mkdir(parents=True, exist_ok=True)
    meta_sidecar = args.drift_out.with_suffix(".meta.json")
    meta_sidecar.write_text(
        json.dumps(
            {"stratification": strat_meta, "hp_threshold_internal": HP_INTERNAL_THRESHOLD},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with args.drift_out.open("w", encoding="utf-8") as out:
        for gid in gids:
            meta = by_cat.get(gid)
            if meta is None:
                rec = {"games_id": gid, "oracle_error": "missing catalog row"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            if not catalog_row_has_both_cos(meta):
                rec = {"games_id": gid, "oracle_error": "catalog incomplete cos"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            mid = _meta_int(meta, "map_id")
            if mid not in std_maps:
                rec = {"games_id": gid, "oracle_error": f"map_id {mid} not in std pool"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            rec = _drill_one_clean(gid, meta, args.zips_dir, args.map_pool, args.maps_dir)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps({"wrote_drift": str(args.drift_out), "n_games": len(gids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
