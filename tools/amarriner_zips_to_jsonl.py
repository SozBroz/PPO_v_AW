#!/usr/bin/env python3
"""
Ingest ``replays/amarriner_gl/{games_id}.zip`` using catalog metadata and append
``human_demos.jsonl`` rows for ``scripts/train_bc.py``.

Optionally prepend lines from another JSONL (e.g. your Andy-mirror traces converted
to rows) so human demos appear first.

Examples::

  python tools/amarriner_zips_to_jsonl.py --replays-dir replays/amarriner_gl \\
    --out data/amarriner_bc_rows.jsonl --manifest data/amarriner_ingest_failures.jsonl

  python tools/amarriner_zips_to_jsonl.py --map-id 123858 --prepend-jsonl data/my_andy.jsonl \\
    --out data/merged_bc_misery.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
REPLAYS_DEFAULT = ROOT / "replays" / "amarriner_gl"


def _load_catalog(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _filter_games(
    games: dict[str, Any],
    *,
    map_id: Optional[int],
    co_p0_id: Optional[int],
    co_p1_id: Optional[int],
    tier: Optional[str],
    mirror_andy: bool,
    require_zip: Path,
    max_games: Optional[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _k, g in games.items():
        if not isinstance(g, dict):
            continue
        gid = int(g["games_id"])
        if map_id is not None and int(g.get("map_id", -1)) != map_id:
            continue
        if tier is not None and str(g.get("tier", "")) != tier:
            continue
        p0 = int(g.get("co_p0_id", -1))
        p1 = int(g.get("co_p1_id", -1))
        if mirror_andy and (p0 != 1 or p1 != 1):
            continue
        if co_p0_id is not None and p0 != co_p0_id:
            continue
        if co_p1_id is not None and p1 != co_p1_id:
            continue
        zpath = require_zip / f"{gid}.zip"
        if not zpath.is_file() or zpath.stat().st_size == 0:
            continue
        rows.append(g)
    rows.sort(key=lambda x: int(x["games_id"]))
    if max_games is not None:
        rows = rows[: max(0, max_games)]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    ap.add_argument("--replays-dir", type=Path, default=REPLAYS_DEFAULT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prepend-jsonl", type=Path, default=None, help="Copy lines first")
    ap.add_argument("--map-id", type=int, default=None)
    ap.add_argument("--co-p0-id", type=int, default=None)
    ap.add_argument("--co-p1-id", type=int, default=None)
    ap.add_argument("--tier", type=str, default=None)
    ap.add_argument("--mirror-andy", action="store_true")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument(
        "--include-move",
        action="store_true",
        help="Pass through to oracle ingest (usually leave off)",
    )
    ap.add_argument("--map-pool", type=Path, default=ROOT / "data" / "gl_map_pool.json")
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "data" / "maps")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Append JSON lines: games_id, error, map_id, tier",
    )
    args = ap.parse_args()

    if not args.catalog.is_file():
        print(f"[ingest] missing catalog {args.catalog}", file=sys.stderr)
        return 1

    data = _load_catalog(args.catalog)
    games = data.get("games") or {}
    selected = _filter_games(
        games,
        map_id=args.map_id,
        co_p0_id=args.co_p0_id,
        co_p1_id=args.co_p1_id,
        tier=args.tier,
        mirror_andy=args.mirror_andy,
        require_zip=args.replays_dir,
        max_games=args.max_games,
    )
    print(f"[ingest] games with zip on disk (after filters): {len(selected)}")

    if args.prepend_jsonl is not None and not args.prepend_jsonl.is_file():
        print(f"[ingest] missing prepend {args.prepend_jsonl}", file=sys.stderr)
        return 1

    from tools.human_demo_rows import collect_demo_rows_from_oracle_zip

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_pre = 0
    manifest_f = None
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_f = open(args.manifest, "a", encoding="utf-8")

    total_oracle = 0
    failed = 0
    try:
        with open(args.out, "w", encoding="utf-8") as out_f:
            if args.prepend_jsonl is not None:
                with open(args.prepend_jsonl, encoding="utf-8") as pre:
                    for line in pre:
                        if line.strip():
                            out_f.write(line if line.endswith("\n") else line + "\n")
                            n_pre += 1
                print(f"[ingest] prepended {n_pre} lines from {args.prepend_jsonl}")

            for g in selected:
                gid = int(g["games_id"])
                zpath = args.replays_dir / f"{gid}.zip"
                map_id = int(g["map_id"])
                co0 = int(g["co_p0_id"])
                co1 = int(g["co_p1_id"])
                tier = str(g["tier"])
                try:
                    rows = collect_demo_rows_from_oracle_zip(
                        zpath,
                        map_pool=args.map_pool,
                        maps_dir=args.maps_dir,
                        map_id=map_id,
                        co0=co0,
                        co1=co1,
                        tier_name=tier,
                        session_prefix=f"amarriner_{gid}",
                        include_move_stage=args.include_move,
                    )
                except Exception as e:
                    failed += 1
                    msg = f"{type(e).__name__}: {e}"
                    print(f"[ingest] FAIL games_id={gid} {msg}")
                    if manifest_f:
                        manifest_f.write(
                            json.dumps(
                                {
                                    "games_id": gid,
                                    "error": msg,
                                    "map_id": map_id,
                                    "tier": tier,
                                }
                            )
                            + "\n"
                        )
                        manifest_f.flush()
                    continue
                for row in rows:
                    out_f.write(json.dumps(row) + "\n")
                    total_oracle += 1
                print(f"[ingest] games_id={gid} rows={len(rows)} (cumulative oracle rows={total_oracle})")
    finally:
        if manifest_f:
            manifest_f.close()

    print(
        f"[ingest] done prepend_lines={n_pre} oracle_rows_written={total_oracle} "
        f"failed_games={failed} -> {args.out}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
