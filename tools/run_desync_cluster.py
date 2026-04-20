"""
Run ``desync_audit.py`` then ``cluster_desync_register.py`` in one step.

- Writes a **dated** register by default: ``logs/desync_register_YYYYMMDD.jsonl``
  (optional ``--tag`` maps to ``desync_register_YYYYMMDD_<tag>.jsonl`` for named snapshots).
- Writes ``logs/desync_clusters.json`` (override with ``--clusters-json``; JSON maps subtype to game id lists).
- Optionally regenerates ``docs/desync_bug_tracker.md`` (``--update-bug-tracker``).

For CI / PR smoke, use ``--pr-smoke`` (implies ``--max-games 25`` and ``--tag ci_smoke``
unless you override with explicit ``--max-games`` / ``--register``). Forward
``--max-games N`` alone for a custom cap.

Examples (repo root)::

  python tools/run_desync_cluster.py
  python tools/run_desync_cluster.py --max-games 20
  python tools/run_desync_cluster.py --pr-smoke
  python tools/run_desync_cluster.py --tag golden --update-bug-tracker
  python tools/run_desync_cluster.py --skip-audit --register logs/desync_register_20260420.jsonl
  python tools/run_desync_cluster.py --skip-audit --register logs/desync_register_YYYYMMDD.jsonl --baseline logs/baselines/desync_register_20260420.jsonl --update-bug-tracker
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.paths import LOGS_DIR, ensure_logs_dir  # noqa: E402


def _default_register_path(tag: str | None) -> Path:
    d = date.today().strftime("%Y%m%d")
    if tag:
        return LOGS_DIR / f"desync_register_{d}_{tag}.jsonl"
    return LOGS_DIR / f"desync_register_{d}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--register",
        type=Path,
        default=None,
        help="Audit output JSONL. Default: logs/desync_register_YYYYMMDD[ _TAG].jsonl",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        metavar="LABEL",
        help="Suffix dated register filename (e.g. golden, ci) before .jsonl",
    )
    ap.add_argument(
        "--clusters-json",
        type=Path,
        default=LOGS_DIR / "desync_clusters.json",
        help="Subtype to games_id JSON (default: logs/desync_clusters.json)",
    )
    ap.add_argument(
        "--update-bug-tracker",
        action="store_true",
        help="Also write docs/desync_bug_tracker.md from the new register",
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional older register JSONL for cluster progress section",
    )
    ap.add_argument(
        "--skip-audit",
        action="store_true",
        help="Only run clustering; --register must exist",
    )
    # Forwarded to tools/desync_audit.py
    ap.add_argument("--catalog", type=Path, default=None)
    ap.add_argument("--zips-dir", type=Path, default=None)
    ap.add_argument("--map-pool", type=Path, default=None)
    ap.add_argument("--maps-dir", type=Path, default=None)
    ap.add_argument("--games-id", type=int, action="append", default=None)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument(
        "--from-bottom",
        action="store_true",
        help="With --max-games, audit highest games_id zips first",
    )
    ap.add_argument(
        "--print-traceback",
        action="store_true",
        help="Forward to desync_audit (engine_bug tracebacks)",
    )
    ap.add_argument(
        "--pr-smoke",
        action="store_true",
        help="Shorthand for PR checks: --max-games 25 and --tag ci_smoke (not applied if "
        "--register is set; --max-games still wins if passed explicitly)",
    )
    args = ap.parse_args()

    if args.pr_smoke:
        if args.max_games is None:
            args.max_games = 25
        if args.register is None and args.tag is None:
            args.tag = "ci_smoke"

    if args.skip_audit:
        if args.register is None:
            print("[run_desync_cluster] --skip-audit requires --register", file=sys.stderr)
            return 2
        register_path = args.register.resolve()
        if not register_path.is_file():
            print(f"[run_desync_cluster] missing register: {register_path}", file=sys.stderr)
            return 1
    else:
        ensure_logs_dir()
        register_path = (
            args.register.resolve()
            if args.register is not None
            else _default_register_path(args.tag).resolve()
        )
        register_path.parent.mkdir(parents=True, exist_ok=True)

        audit_cmd: list[str] = [
            sys.executable,
            str(ROOT / "tools" / "desync_audit.py"),
            "--register",
            str(register_path),
        ]
        if args.catalog is not None:
            audit_cmd += ["--catalog", str(args.catalog)]
        if args.zips_dir is not None:
            audit_cmd += ["--zips-dir", str(args.zips_dir)]
        if args.map_pool is not None:
            audit_cmd += ["--map-pool", str(args.map_pool)]
        if args.maps_dir is not None:
            audit_cmd += ["--maps-dir", str(args.maps_dir)]
        if args.games_id:
            for gid in args.games_id:
                audit_cmd += ["--games-id", str(gid)]
        if args.max_games is not None:
            audit_cmd += ["--max-games", str(args.max_games)]
        if args.from_bottom:
            audit_cmd.append("--from-bottom")
        if args.print_traceback:
            audit_cmd.append("--print-traceback")

        print("[run_desync_cluster] running desync_audit ...", flush=True)
        r = subprocess.run(audit_cmd, cwd=ROOT)
        if r.returncode != 0:
            return r.returncode
        if not register_path.is_file():
            print(
                "[run_desync_cluster] no register file (desync_audit matched no zips); "
                "skipping cluster_desync_register",
                flush=True,
            )
            return 0

    cluster_cmd: list[str] = [
        sys.executable,
        str(ROOT / "tools" / "cluster_desync_register.py"),
        "--register",
        str(register_path),
        "--json",
        str(args.clusters_json),
    ]
    if args.update_bug_tracker:
        cluster_cmd += ["--markdown", str(ROOT / "docs" / "desync_bug_tracker.md")]
    if args.baseline is not None:
        cluster_cmd += ["--baseline", str(args.baseline)]

    print("[run_desync_cluster] running cluster_desync_register ...", flush=True)
    r2 = subprocess.run(cluster_cmd, cwd=ROOT)
    if r2.returncode != 0:
        return r2.returncode

    print(f"[run_desync_cluster] register: {register_path}")
    print(f"[run_desync_cluster] clusters: {args.clusters_json.resolve()}")
    if args.update_bug_tracker:
        print(f"[run_desync_cluster] tracker: {ROOT / 'docs' / 'desync_bug_tracker.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
