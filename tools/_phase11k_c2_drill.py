#!/usr/bin/env python3
"""
Phase 11K-C2-DRILL — per-game End-boundary funds drill for cluster C2 games.

READ-ONLY: imports engine/oracle; writes logs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import make_initial_state  # noqa: E402
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

SUMMARY_CSV = ROOT / "logs" / "phase11k_drift_summary.csv"
JSONL = ROOT / "logs" / "phase11k_drift_data.jsonl"
CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
CO_DATA = ROOT / "data" / "co_data.json"


def _meta_int(meta: dict[str, Any], key: str, default: int = -1) -> int:
    v = meta.get(key, default)
    if v is None:
        return default
    return int(v)


def _php_funds_by_seat(
    php_frame: dict[str, Any], awbw_to_engine: dict[int, int]
) -> dict[int, int]:
    out = {0: 0, 1: 0}
    players = php_frame.get("players") or {}
    for _k, pl in players.items():
        if not isinstance(pl, dict):
            continue
        pid = int(pl["id"])
        eng = awbw_to_engine[pid]
        out[eng] = int(pl.get("funds", 0) or 0)
    return out


def _last_meaningful_action(actions: list[dict[str, Any]]) -> str:
    """Last non-End action in envelope, else End."""
    for obj in reversed(actions):
        a = str(obj.get("action") or "?")
        if a != "End":
            return a
    return "End" if actions else "EMPTY"


def _parse_fund_deltas_from_mismatch_lines(lines: list[str]) -> dict[str, Any]:
    """Extract engine/php ints from 'P0 funds engine=... php_snapshot=...' lines."""
    out: dict[str, Any] = {"P0": None, "P1": None}
    pat = re.compile(
        r"P([01]) funds engine=(\d+) php_snapshot=(\d+).*awbw_players_id=(\d+)"
    )
    for ln in lines or []:
        m = pat.search(ln)
        if not m:
            continue
        eng = int(m.group(2))
        php = int(m.group(3))
        out[f"P{m.group(1)}"] = {
            "engine": eng,
            "php": php,
            "delta_engine_minus_php": eng - php,
        }
    return out


def drill_c2_boundary(
    games_id: int,
    meta: dict[str, Any],
    first_step: int,
) -> dict[str, Any]:
    """
    At first mismatch step_i, report:
    - envelope meta, active CO, last meaningful action
    - PHP vs engine funds at frame[step_i] (pre-envelope) vs frame[step_i+1] (post)
    - deltas
    """
    random.seed(_seed_for_game(CANONICAL_SEED, games_id))
    zip_path = ZIPS / f"{games_id}.zip"
    mid = _meta_int(meta, "map_id")
    co0, co1 = pair_catalog_cos_ids(meta)
    tier_name = str(meta.get("tier") or "T2")

    err: Optional[str] = None
    if not zip_path.is_file():
        return {"games_id": games_id, "error": "missing zip"}

    try:
        frames = load_replay(zip_path)
    except Exception as e:
        return {"games_id": games_id, "error": f"load_replay: {e}"}

    envs = parse_p_envelopes_from_zip(zip_path)
    if not envs or first_step < 0 or first_step >= len(envs):
        return {"games_id": games_id, "error": "bad envs or first_step"}

    try:
        awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    except Exception as e:
        return {"games_id": games_id, "error": str(e)}

    map_data = load_map(mid, MAP_POOL, MAPS_DIR)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data,
        co0,
        co1,
        starting_funds=0,
        tier_name=tier_name,
        replay_first_mover=first_mover,
    )

    # Match _phase11k_drill._drill_one_clean: apply envelopes 0..first_step inclusive.
    pid, day, actions = envs[first_step]
    funds_before_envelope = _php_funds_by_seat(frames[first_step], awbw_to_engine)
    eng_before = {0: int(state.funds[0]), 1: int(state.funds[1])}

    for step_i in range(first_step + 1):
        p2, d2, act2 = envs[step_i]
        for obj in act2:
            try:
                apply_oracle_action_json(
                    state,
                    obj,
                    awbw_to_engine,
                    envelope_awbw_player_id=p2,
                )
            except UnsupportedOracleAction as e:
                return {
                    "games_id": games_id,
                    "first_step": first_step,
                    "failed_at_substep_step_i": step_i,
                    "error": f"UnsupportedOracleAction: {e}",
                }
            except Exception as e:
                return {
                    "games_id": games_id,
                    "first_step": first_step,
                    "failed_at_substep_step_i": step_i,
                    "error": f"{type(e).__name__}: {e}",
                }
            if state.done:
                return {
                    "games_id": games_id,
                    "first_step": first_step,
                    "error": "replay_truncated_before_mismatch",
                }

    snap_i = first_step + 1
    if snap_i >= len(frames):
        return {"games_id": games_id, "first_step": first_step, "error": "no post-envelope frame"}

    php_after = _php_funds_by_seat(frames[snap_i], awbw_to_engine)
    eng_after = {0: int(state.funds[0]), 1: int(state.funds[1])}

    co_active = int(state.co_states[int(state.active_player)].co_id)
    rec = {
        "games_id": games_id,
        "first_step": first_step,
        "envelope_awbw_player_id": int(pid),
        "envelope_day_field": int(day),
        "engine_turn_after": int(state.turn),
        "engine_active_player_after": int(state.active_player),
        "active_co_id_at_boundary": co_active,
        "last_action_in_envelope": str(actions[-1].get("action") if actions else "?"),
        "last_meaningful_action": _last_meaningful_action(actions),
        "funds_php_before_envelope": funds_before_envelope,
        "funds_engine_before_envelope": eng_before,
        "funds_php_after_envelope": php_after,
        "funds_engine_after_envelope": eng_after,
        "php_delta_by_seat": {
            s: php_after[s] - funds_before_envelope[s] for s in (0, 1)
        },
        "engine_delta_by_seat": {
            s: eng_after[s] - eng_before[s] for s in (0, 1)
        },
    }
    return rec


def load_catalog() -> dict[int, dict[str, Any]]:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id: dict[int, dict[str, Any]] = {}
    for _k, g in games.items():
        if isinstance(g, dict) and "games_id" in g:
            by_id[int(g["games_id"])] = g
    return by_id


def load_co_names() -> dict[int, str]:
    data = json.loads(CO_DATA.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for row in (data.get("cos") or {}).values():
        if isinstance(row, dict) and "id" in row:
            out[int(row["id"])] = str(row.get("name") or "?")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="max C2 games to drill")
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=ROOT / "logs" / "phase11k_c2_drill.jsonl",
    )
    ap.add_argument("--out-csv", type=Path, default=ROOT / "logs" / "phase11k_c2_drill.csv")
    ap.add_argument("--gids-file", type=Path, default=ROOT / "logs" / "phase11k_c2_gids.txt")
    args = ap.parse_args()

    by_cat = load_catalog()
    co_names = load_co_names()
    std_maps = gl_std_map_ids(MAP_POOL)

    c2_rows: list[dict[str, str]] = []
    with SUMMARY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cl = row.get("cluster") or ""
            if cl.startswith("C2"):
                c2_rows.append(row)

    args.gids_file.parent.mkdir(parents=True, exist_ok=True)
    gids = [int(r["games_id"]) for r in c2_rows]
    args.gids_file.write_text("\n".join(str(g) for g in gids) + "\n", encoding="utf-8")

    # Enrich from jsonl by games_id
    jsonl_by_gid: dict[int, dict[str, Any]] = {}
    with JSONL.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            jsonl_by_gid[int(o["games_id"])] = o

    drilled: list[dict[str, Any]] = []
    for row in c2_rows[: args.limit]:
        gid = int(row["games_id"])
        j = jsonl_by_gid.get(gid, {})
        first_step = -1
        if row.get("first_step"):
            first_step = int(row["first_step"])
        elif j.get("first_snapshot_mismatch_step") is not None:
            first_step = int(j["first_snapshot_mismatch_step"])
        meta = by_cat.get(gid)
        if meta is None or not catalog_row_has_both_cos(meta):
            drilled.append({"games_id": gid, "error": "catalog"})
            continue
        mid = _meta_int(meta, "map_id")
        if mid not in std_maps:
            drilled.append({"games_id": gid, "error": f"map {mid} not in pool"})
            continue

        mm_lines = j.get("first_mismatch_lines") or []
        fund_parse = _parse_fund_deltas_from_mismatch_lines(mm_lines)

        d = drill_c2_boundary(gid, meta, first_step)
        d["tier"] = row.get("tier")
        d["map_id"] = mid
        d["co_p0_id"] = int(row["co_p0_id"])
        d["co_p1_id"] = int(row["co_p1_id"])
        d["active_co_id_summary"] = int(row["active_co_id"]) if row.get("active_co_id") else None
        d["mismatch_fund_snapshot"] = fund_parse
        d["first_mismatch_line0"] = mm_lines[0] if mm_lines else None

        # Magnitude: max abs drift on mismatched seat from line0
        mag_bucket = "?"
        if mm_lines:
            m0 = mm_lines[0]
            pr = re.search(r"engine=(\d+) php_snapshot=(\d+)", m0)
            if pr:
                delta = abs(int(pr.group(1)) - int(pr.group(2)))
                if delta < 100:
                    mag_bucket = "small"
                elif delta <= 1000:
                    mag_bucket = "medium"
                else:
                    mag_bucket = "large"

        d["magnitude_bucket"] = mag_bucket
        d["active_co_name"] = co_names.get(d.get("active_co_id_at_boundary") or -1, "?")

        # Sign: engine vs php for first seat in mismatch line
        sign = "?"
        if mm_lines and "engine=" in mm_lines[0]:
            pr = re.search(r"P([01]) funds engine=(\d+) php_snapshot=(\d+)", mm_lines[0])
            if pr:
                eng, php = int(pr.group(2)), int(pr.group(3))
                if eng > php:
                    sign = "engine_gt_php"
                elif eng < php:
                    sign = "engine_lt_php"
                else:
                    sign = "tie"

        d["drift_sign"] = sign
        drilled.append(d)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for d in drilled:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")

    # CSV flatten
    fieldnames = [
        "games_id",
        "tier",
        "map_id",
        "first_step",
        "active_co_id_at_boundary",
        "active_co_name",
        "envelope_day_field",
        "engine_turn_after",
        "last_meaningful_action",
        "drift_sign",
        "magnitude_bucket",
        "php_delta_p0",
        "php_delta_p1",
        "engine_delta_p0",
        "engine_delta_p1",
        "first_mismatch_line0",
        "error",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for d in drilled:
            row = {k: "" for k in fieldnames}
            row["games_id"] = d.get("games_id", "")
            row["tier"] = d.get("tier", "")
            row["map_id"] = d.get("map_id", "")
            row["first_step"] = d.get("first_step", "")
            row["active_co_id_at_boundary"] = d.get("active_co_id_at_boundary", "")
            row["active_co_name"] = d.get("active_co_name", "")
            row["envelope_day_field"] = d.get("envelope_day_field", "")
            row["engine_turn_after"] = d.get("engine_turn_after", "")
            row["last_meaningful_action"] = d.get("last_meaningful_action", "")
            row["drift_sign"] = d.get("drift_sign", "")
            row["magnitude_bucket"] = d.get("magnitude_bucket", "")
            pd = d.get("php_delta_by_seat") or {}
            ed = d.get("engine_delta_by_seat") or {}
            row["php_delta_p0"] = pd.get(0, "")
            row["php_delta_p1"] = pd.get(1, "")
            row["engine_delta_p0"] = ed.get(0, "")
            row["engine_delta_p1"] = ed.get(1, "")
            row["first_mismatch_line0"] = d.get("first_mismatch_line0", "")
            row["error"] = d.get("error", "")
            w.writerow(row)

    print(json.dumps({"c2_total_in_csv": len(c2_rows), "drilled": len(drilled), "wrote": str(args.out_jsonl)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
