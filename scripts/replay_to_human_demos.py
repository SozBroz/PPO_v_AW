#!/usr/bin/env python3
"""
Convert a ``*.trace.json`` (preferred) or an AWBW Replay Player ``.zip`` into
``human_demos.jsonl`` rows for ``scripts/train_bc.py``.

Trace JSON matches the engine step-for-step and is the most reliable source.
Oracle zip ingest supports Move / Build / Fire / End as emitted by this repo's
exporter (round-trips with ``test_oracle_zip_replay``); live-site zips may hit
unsupported actions until mappings are extended.

**Manifest mode** (``--manifest``): each line is a JSON object (e.g. from
``tools/fetch_awbw_opening_sources.py``). For each row, either:

* ``trace_path`` — path to a ``*.trace.json`` (if present), or
* ``zip_path`` — AWBW replay zip; metadata via ``map_id`` / ``co0`` / ``co1`` /
  ``tier`` in the row, or infer from the zip's first PHP snapshot
  (``tools.human_demo_rows.infer_training_meta_from_awbw_zip``).

Examples::

  python scripts/replay_to_human_demos.py --trace-json replays/272176.trace.json \\
    --out data/amarriner_bc_misery_andy.jsonl

  python scripts/replay_to_human_demos.py --oracle-zip replays/272176.zip \\
    --map-id 123858 --co0 1 --co1 1 --tier T3 --out data/from_zip.jsonl

  python scripts/replay_to_human_demos.py --manifest data/human_openings/raw/manifest.jsonl \\
    --manifest-base-dir data/human_openings/raw --opening-only --both-seats \\
    --max-days-from-manifest --out data/human_openings/demos/opening_demos.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _max_turn_from_row(row: dict, default: int = 5) -> int | None:
    """``latest`` / null => no cap (None)."""
    d = row.get("requested_days")
    if d is None:
        if row.get("latest_horizon"):
            return None
        return int(default)
    if isinstance(d, str) and d.strip().lower() == "latest":
        return None
    return int(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace-json", type=Path, help="Path to *.trace.json")
    src.add_argument("--oracle-zip", type=Path, help="Path to AWBW replay .zip")
    src.add_argument(
        "--manifest",
        type=Path,
        help="JSONL manifest (``tools/fetch_awbw_opening_sources.py``)",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output .jsonl path")
    ap.add_argument("--session-prefix", type=str, default="replay_ingest")
    ap.add_argument(
        "--include-move",
        action="store_true",
        help="Keep MOVE-stage rows (flat index does not encode destination; see docs/play_ui.md)",
    )
    ap.add_argument("--map-pool", type=Path, default=ROOT / "data" / "gl_map_pool.json")
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "data" / "maps")
    ap.add_argument("--map-id", type=int, default=None, help="Required with --oracle-zip")
    ap.add_argument("--co0", type=int, default=None)
    ap.add_argument("--co1", type=int, default=None)
    ap.add_argument("--tier", type=str, default=None, help="Tier string, e.g. T3")
    ap.add_argument(
        "--manifest-base-dir",
        type=Path,
        default=None,
        help="Base dir to resolve relative zip_path / trace_path in manifest rows",
    )
    ap.add_argument(
        "--opening-only",
        action="store_true",
        help="Tag rows with opening_segment and apply max-day limits when set",
    )
    ap.add_argument(
        "--both-seats",
        action="store_true",
        help="Include engine rows for P0 and P1 (default: P0 only).",
    )
    ap.add_argument(
        "--max-days-from-manifest",
        action="store_true",
        help="Use each manifest row's requested_days / latest as AWBW day cap (trace) or calendar_turn (zip)",
    )
    ap.add_argument(
        "--validate-engine-replay",
        action="store_true",
        help="For oracle zips, run replay_oracle_zip and fail the row on exception",
    )
    args = ap.parse_args()

    from tools.human_demo_rows import (  # noqa: E402
        collect_demo_rows_from_oracle_zip,
        infer_training_meta_from_awbw_zip,
        iter_demo_rows_from_trace_record,
        write_demo_rows_jsonl,
    )

    seats: tuple[int, ...] = (0, 1) if args.both_seats else (0,)

    if args.manifest is not None:
        base = args.manifest_base_dir or args.manifest.parent
        n_total = 0
        first_write = True
        for line in open(args.manifest, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("fetch_status") not in (None, "ok", "cached"):
                continue
            zrel = row.get("zip_path")
            if not zrel:
                print(f"[replay_to_human_demos] skip game_id={row.get('game_id')!r} (no zip_path)")
                continue
            zpath = Path(zrel) if Path(zrel).is_file() else (base / zrel).resolve()
            gid = int(row.get("game_id", 0) or 0)
            tpath = row.get("trace_path")
            if tpath:
                trace: Path | None = Path(tpath) if Path(tpath).is_file() else (base / tpath)
            else:
                cand = (base / "games" / f"{gid}.trace.json").resolve()
                trace = cand if cand.is_file() else None

            max_turn: int | None = None
            mct: int | None = None
            if args.max_days_from_manifest and args.opening_only:
                max_turn = _max_turn_from_row(row)
                mct = max_turn

            if trace is not None and trace.is_file():
                with open(trace, encoding="utf-8") as f:
                    record = json.load(f)
                n = write_demo_rows_jsonl(
                    iter_demo_rows_from_trace_record(
                        record,
                        map_pool=args.map_pool,
                        maps_dir=args.maps_dir,
                        session_prefix=f"{args.session_prefix}:g{gid}",
                        include_move_stage=args.include_move,
                        seats=seats,
                        max_turn=max_turn,
                        opening_only=bool(args.opening_only),
                        source_game_id=gid,
                    ),
                    args.out,
                    append=not first_write,
                )
                first_write = False
                n_total += n
                print(f"[replay_to_human_demos] trace game_id={gid} rows={n} -> {args.out}")
                continue

            if not zpath.is_file():
                print(f"[replay_to_human_demos] missing zip {zpath!r} game_id={gid}", file=sys.stderr)
                continue

            if args.validate_engine_replay:
                from tools.oracle_zip_replay import replay_oracle_zip  # noqa: E402

                if (
                    row.get("map_id") is not None
                    and row.get("co_p0_id") is not None
                    and row.get("co_p1_id") is not None
                ):
                    meta0 = {
                        "map_id": int(row["map_id"]),
                        "co0": int(row["co_p0_id"]),
                        "co1": int(row["co_p1_id"]),
                        "tier": str(row.get("tier") or "T3"),
                    }
                else:
                    meta0 = infer_training_meta_from_awbw_zip(zpath, map_pool=args.map_pool)
                try:
                    replay_oracle_zip(
                        zpath,
                        map_pool=args.map_pool,
                        maps_dir=args.maps_dir,
                        map_id=int(meta0["map_id"]),
                        co0=int(meta0["co0"]),
                        co1=int(meta0["co1"]),
                        tier_name=str(meta0["tier"]),
                    )
                except Exception as exc:
                    print(
                        f"[replay_to_human_demos] validate fail game_id={gid}: {exc}",
                        file=sys.stderr,
                    )
                    continue

            if (
                row.get("map_id") is not None
                and row.get("co_p0_id") is not None
                and row.get("co_p1_id") is not None
            ):
                meta = {
                    "map_id": int(row["map_id"]),
                    "co0": int(row["co_p0_id"]),
                    "co1": int(row["co_p1_id"]),
                    "tier": str(row.get("tier") or "T3"),
                }
            else:
                meta = infer_training_meta_from_awbw_zip(zpath, map_pool=args.map_pool)
            rows = collect_demo_rows_from_oracle_zip(
                zpath,
                map_pool=args.map_pool,
                maps_dir=args.maps_dir,
                map_id=int(meta["map_id"]),
                co0=int(meta["co0"]),
                co1=int(meta["co1"]),
                tier_name=str(meta.get("tier", "T3")),
                session_prefix=f"{args.session_prefix}:g{gid}",
                include_move_stage=args.include_move,
                seats=seats,
                max_calendar_turn=mct,
                opening_only=bool(args.opening_only),
                source_game_id=gid,
            )
            n = write_demo_rows_jsonl(iter(rows), args.out, append=not first_write)
            first_write = False
            n_total += n
            print(f"[replay_to_human_demos] oracle game_id={gid} rows={n} -> {args.out} (total {n_total})")
        if n_total == 0:
            print("[replay_to_human_demos] no rows written", file=sys.stderr)
            return 1
        return 0

    if args.trace_json is not None:
        with open(args.trace_json, encoding="utf-8") as f:
            record = json.load(f)
        n = write_demo_rows_jsonl(
            iter_demo_rows_from_trace_record(
                record,
                map_pool=args.map_pool,
                maps_dir=args.maps_dir,
                session_prefix=args.session_prefix,
                include_move_stage=args.include_move,
                seats=seats,
            ),
            args.out,
        )
        print(f"[replay_to_human_demos] trace rows={n} -> {args.out}")
        return 0

    assert args.oracle_zip is not None
    if args.map_id is None or args.co0 is None or args.co1 is None or args.tier is None:
        raise SystemExit("--oracle-zip requires --map-id, --co0, --co1, and --tier")
    rows = collect_demo_rows_from_oracle_zip(
        args.oracle_zip,
        map_pool=args.map_pool,
        maps_dir=args.maps_dir,
        map_id=args.map_id,
        co0=args.co0,
        co1=args.co1,
        tier_name=args.tier,
        session_prefix=args.session_prefix,
        include_move_stage=args.include_move,
        seats=seats,
    )
    n = write_demo_rows_jsonl(iter(rows), args.out)
    print(f"[replay_to_human_demos] oracle-zip rows={n} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
