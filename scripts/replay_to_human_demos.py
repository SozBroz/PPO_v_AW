#!/usr/bin/env python3
"""
Convert a ``*.trace.json`` (preferred) or an AWBW Replay Player ``.zip`` into
``human_demos.jsonl`` rows for ``scripts/train_bc.py``.

Trace JSON matches the engine step-for-step and is the most reliable source.
Oracle zip ingest supports Move / Build / Fire / End as emitted by this repo's
exporter (round-trips with ``test_oracle_zip_replay``); live-site zips may hit
unsupported actions until mappings are extended.

Examples::

  python scripts/replay_to_human_demos.py --trace-json replays/272176.trace.json \\
    --out data/amarriner_bc_misery_andy.jsonl

  python scripts/replay_to_human_demos.py --oracle-zip replays/272176.zip \\
    --map-id 123858 --co0 1 --co1 1 --tier T3 --out data/from_zip.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace-json", type=Path, help="Path to *.trace.json")
    src.add_argument("--oracle-zip", type=Path, help="Path to AWBW replay .zip")
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
    args = ap.parse_args()

    if args.trace_json is not None:
        with open(args.trace_json, encoding="utf-8") as f:
            record = json.load(f)
        from tools.human_demo_rows import iter_demo_rows_from_trace_record, write_demo_rows_jsonl

        n = write_demo_rows_jsonl(
            iter_demo_rows_from_trace_record(
                record,
                map_pool=args.map_pool,
                maps_dir=args.maps_dir,
                session_prefix=args.session_prefix,
                include_move_stage=args.include_move,
            ),
            args.out,
        )
        print(f"[replay_to_human_demos] trace rows={n} -> {args.out}")
        return 0

    assert args.oracle_zip is not None
    if args.map_id is None or args.co0 is None or args.co1 is None or args.tier is None:
        raise SystemExit("--oracle-zip requires --map-id, --co0, --co1, and --tier")
    from tools.human_demo_rows import collect_demo_rows_from_oracle_zip, write_demo_rows_jsonl

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
    )
    n = write_demo_rows_jsonl(iter(rows), args.out)
    print(f"[replay_to_human_demos] oracle-zip rows={n} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
