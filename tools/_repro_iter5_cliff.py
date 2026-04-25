"""
Reproducer harness for train.py FPS cliff at n_envs=6 (Phase 6b).

Captures ``logs/fps_diag.jsonl`` rows and periodic RSS samples while a real
``train.py`` subprocess runs. Does not interpret results — analysis is manual.

Example::

    python tools/_repro_iter5_cliff.py --n-envs 6 --max-iters 8 --out logs/repro_iter5_n6.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.train_launch_env import environ_for_train_subprocess


def _sum_python_descendants_rss_mb(proc: Any) -> float:
    try:
        import psutil  # type: ignore[import]

        p = psutil.Process(proc.pid)
        total = int(p.memory_info().rss)
        for c in p.children(recursive=True):
            try:
                if "python" not in (c.name() or "").lower():
                    continue
                total += int(c.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return float(total) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _main_rss_mb(proc: Any) -> float:
    try:
        import psutil  # type: ignore[import]

        return float(psutil.Process(proc.pid).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _system_ram_pct() -> float:
    try:
        import psutil  # type: ignore[import]

        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture FPS / RSS diagnostics around train.py iter-5 cliff.")
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-iters", type=int, default=8, help="Stop after this many rollouts observed in fps_diag (approx).")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("logs/repro_iter5_n6.json"),
        help="Summary JSON path (default logs/repro_iter5_n6.json under repo).",
    )
    ap.add_argument("--map-id", type=int, default=123858)
    ap.add_argument("--tier", type=str, default="T3")
    ap.add_argument("--co-p0", type=int, default=1)
    ap.add_argument("--co-p1", type=int, default=1)
    ap.add_argument("--machine-id", type=str, default="pc-b")
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    args = ap.parse_args()

    os.environ["AWBW_TRACK_PER_WORKER_TIMES"] = "1"
    os.environ["AWBW_MACHINE_ID"] = args.machine_id

    out_path = args.out
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psutil_log = out_path.with_name(out_path.stem + "_psutil.jsonl")

    import tempfile

    ckpt_dir = Path(tempfile.mkdtemp(prefix="awbw_repro_ckpt_", dir=str(REPO_ROOT / "logs")))

    total_ts = int(args.max_iters * args.n_steps * args.n_envs)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--n-envs",
        str(args.n_envs),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--cold-opponent",
        "random",
        "--map-id",
        str(args.map_id),
        "--tier",
        args.tier,
        "--co-p0",
        str(args.co_p0),
        "--co-p1",
        str(args.co_p1),
        "--curriculum-broad-prob",
        "0",
        "--save-every",
        "999999",
        "--checkpoint-dir",
        str(ckpt_dir),
        "--iters",
        str(total_ts),
    ]

    fps_diag_path = REPO_ROOT / "logs" / "fps_diag.jsonl"
    stop_evt = threading.Event()
    seen_diag_lines: list[str] = []
    lock = threading.Lock()
    peak_ram_pct = 0.0

    def poll_fps_diag() -> None:
        nonlocal seen_diag_lines
        pos = 0
        if fps_diag_path.is_file():
            pos = fps_diag_path.stat().st_size
        while not stop_evt.is_set():
            time.sleep(5.0)
            if not fps_diag_path.is_file():
                continue
            try:
                with open(fps_diag_path, "r", encoding="utf-8") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                if chunk:
                    with lock:
                        for line in chunk.splitlines():
                            line = line.strip()
                            if line:
                                seen_diag_lines.append(line)
            except Exception:
                continue

    def poll_psutil(proc_holder: list) -> None:
        nonlocal peak_ram_pct
        while not stop_evt.is_set():
            time.sleep(2.0)
            proc = proc_holder[0]
            if proc is None:
                continue
            try:
                rec = {
                    "t": time.time(),
                    "train_pid": proc.pid,
                    "main_rss_mb": _main_rss_mb(proc),
                    "sum_python_rss_mb": _sum_python_descendants_rss_mb(proc),
                    "system_ram_used_pct": _system_ram_pct(),
                }
                peak_ram_pct = max(peak_ram_pct, rec["system_ram_used_pct"])
                with open(psutil_log, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps(rec) + "\n")
            except Exception:
                continue

    proc_holder: list[Any] = [None]
    t_diag = threading.Thread(target=poll_fps_diag, daemon=True)
    t_ps = threading.Thread(target=poll_psutil, args=(proc_holder,), daemon=True)
    t_diag.start()
    t_ps.start()

    started_at = time.time()
    t_start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=environ_for_train_subprocess(),
    )
    proc_holder[0] = proc
    exit_code: int | None = None
    try:
        while True:
            if proc.poll() is not None:
                exit_code = int(proc.returncode or 0)
                break
            if time.perf_counter() - t_start > args.timeout_s:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                exit_code = -9
                break
            with lock:
                n_rollouts = len(seen_diag_lines)
                last_iter = 0
                if seen_diag_lines:
                    try:
                        last_iter = int(json.loads(seen_diag_lines[-1]).get("iteration", 0))
                    except Exception:
                        pass
            if n_rollouts >= args.max_iters or last_iter >= args.max_iters:
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
                exit_code = int(proc.returncode or 0)
                break
            time.sleep(0.5)
    finally:
        stop_evt.set()
        t_diag.join(timeout=2.0)
        t_ps.join(timeout=2.0)

    ended_at = time.time()
    last_env_collect = None
    last_main_rss = None
    last_sum_worker = None
    last_iteration = None
    with lock:
        num_iter_obs = len(seen_diag_lines)
        if seen_diag_lines:
            try:
                last = json.loads(seen_diag_lines[-1])
                last_iteration = last.get("iteration")
                last_env_collect = last.get("env_collect_s")
                last_main_rss = last.get("main_proc_rss_mb")
                last_sum_worker = last.get("sum_worker_rss_mb")
            except Exception:
                pass

    summary = {
        "started_at": started_at,
        "ended_at": ended_at,
        "exit_code": exit_code,
        "last_iteration": last_iteration,
        "last_env_collect_s": last_env_collect,
        "last_main_rss_mb": last_main_rss,
        "last_sum_worker_rss_mb": last_sum_worker,
        "peak_system_ram_pct": peak_ram_pct,
        "num_iterations_observed": num_iter_obs,
        "fps_diag_lines_captured": num_iter_obs,
        "train_command": cmd,
        "checkpoint_dir": str(ckpt_dir),
        "psutil_jsonl": str(psutil_log),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[repro] Wrote summary -> {out_path}")
    print(f"[repro] psutil samples -> {psutil_log}")
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
