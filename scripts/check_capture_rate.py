"""Rolling P0 capture-rate kill-switch watchdog.

Tails ``logs/game_log.jsonl`` and computes a rolling-N P0 capture rate (mean
captures per game). Prints a WARNING / ERROR line if the rate drops below the
configured floor after the warmup window — the regression detector that the
plan ``p0-capture-architecture-fix`` requires.

Designed to be cheap and dependency-free: stdlib only, polls the file at a
fixed interval. Exit 0 unless ``--exit-on-fail`` is set, in which case the
process exits non-zero the first time the gate fails after warmup.

Usage::

    python scripts/check_capture_rate.py
    python scripts/check_capture_rate.py --window 100 --floor 1.0 --warmup 500

Reads the same ``GAME_LOG_PATH`` as the training loop so you do not need to
synchronise anything. Multiple watchers are safe — file is opened read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.paths import GAME_LOG_PATH


def _iter_new_lines(path: Path, last_pos: int) -> tuple[list[str], int]:
    """Return (new_lines, new_byte_position). Empty list if file unchanged."""
    if not path.exists():
        return [], last_pos
    size = path.stat().st_size
    if size < last_pos:
        # File rotated/truncated; restart from the top.
        last_pos = 0
    if size == last_pos:
        return [], last_pos
    with open(path, "rb") as fh:
        fh.seek(last_pos)
        chunk = fh.read(size - last_pos)
    last_pos = size
    text = chunk.decode("utf-8", errors="replace")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return lines, last_pos


def _parse_record(line: str) -> dict | None:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(rec, dict):
        return None
    return rec


def _extract_caps(rec: dict) -> tuple[int | None, int | None, int | None]:
    """Return (game_id, p0_captures, p1_captures) or Nones if missing."""
    gid = rec.get("game_id")
    p0 = rec.get("captures_completed_p0")
    p1 = rec.get("captures_completed_p1")
    if not isinstance(gid, int) or not isinstance(p0, int) or not isinstance(p1, int):
        return None, None, None
    return gid, p0, p1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--window", type=int, default=100,
        help="Rolling-window game count for the capture-rate average (default: 100)",
    )
    p.add_argument(
        "--floor", type=float, default=1.0,
        help="Mean P0 captures-per-game below which to alert (default: 1.0)",
    )
    p.add_argument(
        "--warmup", type=int, default=500,
        help="Number of games before the floor gate becomes active (default: 500)",
    )
    p.add_argument(
        "--poll-interval-s", type=float, default=10.0,
        help="Seconds between log-tail polls (default: 10s)",
    )
    p.add_argument(
        "--print-every", type=int, default=25,
        help="Print a status line every N games (default: 25)",
    )
    p.add_argument(
        "--exit-on-fail", action="store_true",
        help="Exit with non-zero status the first time the gate fails after warmup",
    )
    p.add_argument(
        "--log-path", type=Path, default=GAME_LOG_PATH,
        help=f"Override game_log.jsonl path (default: {GAME_LOG_PATH})",
    )
    args = p.parse_args()

    print(
        f"[capture-watch] tail={args.log_path} window={args.window} "
        f"floor={args.floor} warmup={args.warmup} poll={args.poll_interval_s}s",
        flush=True,
    )

    seen: set[int] = set()
    p0_window: deque[int] = deque(maxlen=args.window)
    p1_window: deque[int] = deque(maxlen=args.window)
    games_total = 0
    last_pos = 0
    exit_code = 0

    while True:
        try:
            lines, last_pos = _iter_new_lines(args.log_path, last_pos)
        except OSError as exc:
            print(f"[capture-watch] read error: {exc}", flush=True)
            time.sleep(args.poll_interval_s)
            continue

        for line in lines:
            rec = _parse_record(line)
            if rec is None:
                continue
            gid, p0c, p1c = _extract_caps(rec)
            if gid is None or gid in seen:
                continue
            seen.add(gid)
            p0_window.append(p0c)
            p1_window.append(p1c)
            games_total += 1

            if games_total % args.print_every == 0 or games_total == args.warmup:
                avg_p0 = sum(p0_window) / len(p0_window) if p0_window else 0.0
                avg_p1 = sum(p1_window) / len(p1_window) if p1_window else 0.0
                tag = ""
                if games_total >= args.warmup and avg_p0 < args.floor:
                    tag = " WARNING below-floor"
                    if args.exit_on_fail and exit_code == 0:
                        exit_code = 2
                        print(
                            f"[capture-watch] FAIL games={games_total} "
                            f"p0_caps_avg={avg_p0:.2f} < floor={args.floor}",
                            flush=True,
                        )
                        return exit_code
                print(
                    f"[capture-watch] games={games_total} "
                    f"p0_caps_avg={avg_p0:.2f} p1_caps_avg={avg_p1:.2f}"
                    f" (window={len(p0_window)}){tag}",
                    flush=True,
                )

        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    sys.exit(main())
