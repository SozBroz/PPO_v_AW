"""
Phase 9: optional wall-clock training throughput bench (manual / cron — not CI).

Runs ``train.py`` for a fixed duration, then summarizes ``env_steps_per_s_total``
from ``logs/fps_diag.jsonl`` and appends one JSON line to
``logs/bench_train_throughput.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.train_launch_env import environ_for_train_subprocess

from tools.fps_diag_metrics import parse_fps_diag_lines, summarize_fps

TRAIN_PY = REPO_ROOT / "train.py"
LOGS_DIR = REPO_ROOT / "logs"
FPS_DIAG_PATH = LOGS_DIR / "fps_diag.jsonl"
BENCH_LOG_PATH = LOGS_DIR / "bench_train_throughput.jsonl"


def _read_diag_bytes_from_offset(path: Path, start_off: int) -> str:
    if not path.is_file():
        return ""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        sz = fh.tell()
        if sz < start_off:
            fh.seek(0)
        else:
            fh.seek(start_off)
        return fh.read().decode("utf-8", errors="replace")


def _git_sha(repo: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wall-time train.py throughput bench")
    ap.add_argument("--budget-seconds", type=float, default=300.0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--machine-id",
        type=str,
        default=None,
        help=(
            "Forwarded to train.py as --machine-id; subprocess env AWBW_MACHINE_ID "
            "is set to the same value. Default: $AWBW_MACHINE_ID or bench."
        ),
    )
    ap.add_argument(
        "--config-name",
        type=str,
        default="baseline_pcb_n4",
        help="Label stored in the bench row (opaque tag for your experiment grid).",
    )
    args = ap.parse_args(argv)

    if not TRAIN_PY.is_file():
        print(f"error: missing {TRAIN_PY}", file=sys.stderr)
        return 2

    mid = args.machine_id
    if mid is None or not str(mid).strip():
        mid = os.environ.get("AWBW_MACHINE_ID") or "bench"
    mid = str(mid).strip() or "bench"

    diag_off = FPS_DIAG_PATH.stat().st_size if FPS_DIAG_PATH.is_file() else 0

    train_cmd = [
        sys.executable,
        str(TRAIN_PY),
        "--machine-id",
        mid,
        "--iters",
        str(10**12),
        "--n-envs",
        str(args.n_envs),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
    ]

    env = environ_for_train_subprocess()
    env["AWBW_MACHINE_ID"] = mid

    t0 = time.perf_counter()
    exit_code: int | None = None
    timed_out = False
    try:
        proc = subprocess.run(
            train_cmd,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=float(args.budget_seconds),
        )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        if exc.process is not None:
            exc.process.kill()
            try:
                exc.process.wait(timeout=60)
            except (OSError, subprocess.SubprocessError):
                pass
    wall_s = time.perf_counter() - t0

    new_text = _read_diag_bytes_from_offset(FPS_DIAG_PATH, diag_off)
    fps_values = parse_fps_diag_lines(new_text)
    fps_stats = summarize_fps(fps_values)

    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(REPO_ROOT),
        "machine_id": mid,
        "config_name": args.config_name,
        "config": {
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "budget_seconds": args.budget_seconds,
        },
        "train_cmd": train_cmd,
        "train_exit_code": exit_code,
        "timed_out": timed_out,
        "wall_s": round(wall_s, 3),
        "fps_diag_path": str(FPS_DIAG_PATH.relative_to(REPO_ROOT)),
        "fps_env_steps_per_s": fps_stats,
    }
    _append_jsonl(BENCH_LOG_PATH, row)
    print(json.dumps(row, indent=2))
    return 0 if exit_code in (0, 124) else (exit_code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
