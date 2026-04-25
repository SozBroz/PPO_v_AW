"""Batch-smoke ``rl/ai_vs_ai.py`` + trace round-trip ``write_awbw_replay_from_trace``.

Example::

    python tools/smoke_ai_vs_ai_batch.py --runs 10 --max-turns 100 --random

Uses fixed ``--game-id`` ranges so outputs land in ``replays/`` predictably.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=10, help="Number of ai_vs_ai sessions")
    ap.add_argument("--max-turns", type=int, default=100, dest="max_turns")
    ap.add_argument("--random", action="store_true", help="Random policy (no checkpoint)")
    ap.add_argument("--game-id-base", type=int, default=950_000, help="First --game-id")
    ap.add_argument("--seed-base", type=int, default=50_000, help="First --seed offset")
    ap.add_argument("--skip-reexport", action="store_true", help="Only run ai_vs_ai")
    args = ap.parse_args()

    failures: list[tuple[int, str, str]] = []

    for i in range(args.runs):
        gid = args.game_id_base + i
        seed = args.seed_base + i
        cmd = [
            sys.executable,
            str(ROOT / "rl" / "ai_vs_ai.py"),
            "--no-follow-train",
            "--no-open",
            "--max-turns",
            str(args.max_turns),
            "--seed",
            str(seed),
            "--game-id",
            str(gid),
        ]
        if args.random:
            cmd.append("--random")

        print(f"=== run {i + 1}/{args.runs}  game_id={gid}  seed={seed} ===", flush=True)
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
        except subprocess.TimeoutExpired:
            failures.append((i, "timeout", f"game_id={gid}"))
            print(f"TIMEOUT run {i}", flush=True)
            continue

        if r.returncode != 0:
            failures.append((i, "ai_vs_ai", f"exit={r.returncode}\n{r.stdout}\n{r.stderr}"))
            print(f"FAIL ai_vs_ai run {i}", flush=True)
            continue

        if args.skip_reexport:
            print(f"ok run {i + 1} (skip reexport)", flush=True)
            continue

        trace_path = ROOT / "replays" / f"{gid}.trace.json"
        if not trace_path.is_file():
            failures.append((i, "missing_trace", str(trace_path)))
            continue

        record = json.loads(trace_path.read_text(encoding="utf-8"))
        if "full_trace" not in record:
            failures.append((i, "trace_schema", "no full_trace"))
            continue

        from tools.export_awbw_replay import write_awbw_replay_from_trace

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / f"reexport_{gid}.zip"
            try:
                write_awbw_replay_from_trace(record, out, game_id=gid)
            except Exception as exc:
                failures.append((i, "reexport", repr(exc)))
                print(f"FAIL reexport run {i}: {exc!r}", flush=True)
                continue

        print(
            f"ok run {i + 1}  trace_actions={record.get('n_actions_full_trace')}",
            flush=True,
        )

    if failures:
        print("\n--- FAILURES ---", flush=True)
        for idx, kind, msg in failures:
            print(f"[{idx}] {kind}:\n{msg[:6000]}\n", flush=True)
        return 1
    print("\nAll runs + trace round-trip export OK.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
