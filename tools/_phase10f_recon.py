#!/usr/bin/env python3
"""
Phase 10F — silent-drift recon for games classified ``ok`` in desync_register.

Uses :func:`tools.replay_state_diff.run_zip` (PHP snapshot vs engine after each
envelope) with the same per-game RNG seed as :func:`tools.desync_audit._audit_one`
so the engine path matches the audit that produced ``class: ok``.

Does not modify ``desync_audit.py`` or ``oracle_zip_replay.py``.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.amarriner_catalog_cos import catalog_row_has_both_cos, pair_catalog_cos_ids  # noqa: E402
from tools.desync_audit import CANONICAL_SEED, _seed_for_game  # noqa: E402
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402
from tools.replay_state_diff import run_zip  # noqa: E402

REGISTER_OK = ROOT / "logs" / "desync_register_post_phase9.jsonl"
CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS_DEFAULT = ROOT / "replays" / "amarriner_gl"
MAP_POOL_DEFAULT = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR_DEFAULT = ROOT / "data" / "maps"
OUT_JSONL_DEFAULT = ROOT / "logs" / "phase10f_silent_drift.jsonl"


def _meta_int(meta: dict[str, Any], key: str, default: int = -1) -> int:
    v = meta.get(key, default)
    if v is None:
        return default
    return int(v)


def _classify_drift_kind(mismatches: list[str]) -> str:
    """Best-effort label from ``compare_snapshot_to_engine`` strings."""
    if not mismatches:
        return "none"
    blob = " ".join(mismatches[:20])
    if re.search(r"P[01] funds engine=", blob):
        return "funds"
    if "unit tile set mismatch" in blob or "only_in_php" in blob or "only_in_engine" in blob:
        return "position"
    if "hp_bars" in blob:
        return "hp"
    if "type engine=" in blob or "duplicate unit" in blob or "php duplicate" in blob:
        return "structure"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register-ok", type=Path, default=REGISTER_OK)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    ap.add_argument("--zips-dir", type=Path, default=ZIPS_DEFAULT)
    ap.add_argument("--map-pool", type=Path, default=MAP_POOL_DEFAULT)
    ap.add_argument("--maps-dir", type=Path, default=MAPS_DIR_DEFAULT)
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--random-seed", type=int, default=1, help="Seed for sampling ok gids.")
    ap.add_argument("--out-jsonl", type=Path, default=OUT_JSONL_DEFAULT)
    args = ap.parse_args()

    ok_gids: list[int] = []
    if not args.register_ok.is_file():
        print(f"[_phase10f_recon] missing {args.register_ok}", file=sys.stderr)
        return 1
    with args.register_ok.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("class") == "ok":
                ok_gids.append(int(r["games_id"]))

    random.seed(args.random_seed)
    n = min(int(args.sample_size), len(ok_gids))
    sample = sorted(random.sample(ok_gids, n)) if ok_gids else []

    cat = json.loads(args.catalog.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id: dict[int, dict[str, Any]] = {}
    for _k, g in games.items():
        if isinstance(g, dict) and "games_id" in g:
            by_id[int(g["games_id"])] = g

    std_maps = gl_std_map_ids(args.map_pool)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for gid in sample:
        meta = by_id.get(gid)
        zip_path = args.zips_dir / f"{gid}.zip"
        row: dict[str, Any] = {
            "games_id": gid,
            "zip_exists": zip_path.is_file(),
            "map_id": None,
            "tier": None,
            "co_p0_id": None,
            "co_p1_id": None,
            "in_std_pool": None,
            "catalog_complete": None,
            "snapshot_diff_ok": None,
            "pairing": None,
            "aligned": None,
            "first_step_mismatch": None,
            "drift_kind": None,
            "oracle_error": None,
            "step_mismatch_preview": [],
            "initial_mismatch_preview": [],
        }
        if meta is None:
            row["oracle_error"] = "missing catalog row"
            rows.append(row)
            continue
        mid = _meta_int(meta, "map_id")
        row["map_id"] = mid
        row["tier"] = str(meta.get("tier") or "")
        row["co_p0_id"] = _meta_int(meta, "co_p0_id")
        row["co_p1_id"] = _meta_int(meta, "co_p1_id")
        row["in_std_pool"] = mid in std_maps
        if not zip_path.is_file():
            row["oracle_error"] = "zip missing"
            rows.append(row)
            continue
        if not catalog_row_has_both_cos(meta):
            row["catalog_complete"] = False
            row["oracle_error"] = "catalog missing co ids"
            rows.append(row)
            continue
        row["catalog_complete"] = True
        if mid not in std_maps:
            row["oracle_error"] = "map not in GL std pool (skipped in replay_state_diff)"
            rows.append(row)
            continue

        co0, co1 = pair_catalog_cos_ids(meta)
        random.seed(_seed_for_game(CANONICAL_SEED, gid))
        r = run_zip(
            zip_path=zip_path,
            map_pool=args.map_pool,
            maps_dir=args.maps_dir,
            map_id=mid,
            co0=co0,
            co1=co1,
            tier_name=str(meta.get("tier") or "T2"),
            sync_to_php=False,
        )
        row["snapshot_diff_ok"] = r.ok
        row["pairing"] = r.pairing
        row["aligned"] = r.aligned
        row["first_step_mismatch"] = r.first_step_mismatch
        row["oracle_error"] = r.oracle_error
        row["step_mismatch_preview"] = (r.step_mismatches or [])[:12]
        row["initial_mismatch_preview"] = (r.initial_mismatches or [])[:12]
        mm = r.step_mismatches or r.initial_mismatches or []
        row["drift_kind"] = _classify_drift_kind(mm) if not r.ok else "none"
        rows.append(row)

    drift_n = sum(1 for x in rows if x.get("snapshot_diff_ok") is False)
    clean_n = sum(1 for x in rows if x.get("snapshot_diff_ok") is True)
    early_end_clean = sum(
        1
        for x in rows
        if x.get("snapshot_diff_ok") is True
        and x.get("oracle_error")
        and "before zip exhausted" in str(x.get("oracle_error"))
    )
    full_zip_clean = sum(
        1
        for x in rows
        if x.get("snapshot_diff_ok") is True
        and not x.get("oracle_error")
    )
    inconclusive_n = len(rows) - drift_n - clean_n

    summary = {
        "sample_requested": int(args.sample_size),
        "ok_gids_in_register": len(ok_gids),
        "sample_size": len(sample),
        "random_seed": args.random_seed,
        "rng_match_desync_audit": "per-game random.seed(_seed_for_game(CANONICAL_SEED, games_id))",
        "comparator": "tools.replay_state_diff.run_zip (PHP awbwGame snapshots vs engine)",
        "with_drift": drift_n,
        "clean_no_mismatch_before_stop": clean_n,
        "clean_full_zip_compared": full_zip_clean,
        "clean_but_resign_before_exhaust": early_end_clean,
        "inconclusive": inconclusive_n,
    }

    with args.out_jsonl.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"_phase10f_summary": summary}, ensure_ascii=False) + "\n")
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
