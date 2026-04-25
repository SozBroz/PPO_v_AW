"""
Sweep ``--n-envs`` candidates using short ``train.py`` probes and fps_diag medians.

Depends on psutil + stdlib. Report: ``<fleet_dir>/throughput_tune.json`` (atomic write).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from rl.train_launch_env import environ_for_train_subprocess

from tools.fps_diag_metrics import (
    parse_fps_diag_collect_lines,
    parse_fps_diag_lines,
    parse_fps_diag_throughput_values,
    summarize_fps,
)

HYSTERESIS_DELTA = 5.0


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


def _baseline_n_envs_from_proposed(proposed: dict[str, Any]) -> int:
    args = proposed.get("args") or {}
    raw = args.get("--n-envs")
    if raw is None:
        raise KeyError("proposed['args']['--n-envs']")
    return int(raw)


def _machine_id_from_train_argv(argv: list[str]) -> str | None:
    for i, tok in enumerate(argv):
        if tok == "--machine-id" and i + 1 < len(argv):
            m = str(argv[i + 1]).strip()
            return m or None
        if tok.startswith("--machine-id="):
            m = str(tok.split("=", 1)[1]).strip()
            return m or None
    return None


def _n_steps_from_argv(argv: list[str], proposed: dict[str, Any]) -> int:
    for i, tok in enumerate(argv):
        if tok == "--n-steps" and i + 1 < len(argv):
            return int(argv[i + 1])
        if tok.startswith("--n-steps="):
            return int(tok.split("=", 1)[1])
    args = proposed.get("args") or {}
    return int(args.get("--n-steps", 512))


def _argv_with_probe_iters(argv: list[str], iters: int) -> list[str]:
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(argv):
        if argv[i] == "--iters" and i + 1 < len(argv):
            out.extend(["--iters", str(iters)])
            i += 2
            replaced = True
            continue
        if argv[i].startswith("--iters="):
            out.append(f"--iters={iters}")
            i += 1
            replaced = True
            continue
        out.append(argv[i])
        i += 1
    if not replaced:
        out.extend(["--iters", str(iters)])
    return out


def _wait_host_headroom(
    *,
    max_host_ram_pct: float,
    max_host_cpu_pct: float,
    host_wait_s: float,
    log: logging.Logger,
    phase: str,
) -> bool:
    """
    If host is at/above caps, poll every 3s until hysteresis-clear or timeout.
    Returns False if still at/above caps after ``host_wait_s`` (caller aborts).
    """
    ram_clear = max_host_ram_pct - HYSTERESIS_DELTA
    cpu_clear = max_host_cpu_pct - HYSTERESIS_DELTA

    def overloaded() -> bool:
        ram = float(psutil.virtual_memory().percent)
        cpu = float(psutil.cpu_percent(interval=0.25))
        return ram >= max_host_ram_pct or cpu >= max_host_cpu_pct

    def clear() -> bool:
        ram = float(psutil.virtual_memory().percent)
        cpu = float(psutil.cpu_percent(interval=0.25))
        return ram < ram_clear and cpu < cpu_clear

    if not overloaded():
        return True

    log.warning(
        "[%s] Host overloaded (ram/cpu vs max); waiting up to %.0fs for headroom...",
        phase,
        host_wait_s,
    )
    deadline = time.monotonic() + float(host_wait_s)
    while time.monotonic() < deadline:
        time.sleep(3.0)
        if clear():
            log.info("[%s] Host headroom recovered (hysteresis).", phase)
            return True

    if overloaded():
        log.error("[%s] Host still overloaded after wait; aborting tune.", phase)
        return False
    return True


def _lower_probe_priority(proc: subprocess.Popen) -> None:
    if proc.pid is None:
        return
    try:
        root = psutil.Process(proc.pid)
    except psutil.Error:
        return
    try:
        if sys.platform == "win32":
            root.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            for c in root.children(recursive=True):
                try:
                    c.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except psutil.Error:
                    pass
        else:
            try:
                n = int(root.nice())
            except psutil.Error:
                n = 0
            root.nice(min(19, n + 5))
            for c in root.children(recursive=True):
                try:
                    cn = int(c.nice())
                    c.nice(min(19, cn + 5))
                except psutil.Error:
                    pass
    except (psutil.Error, OSError, ValueError):
        pass


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def choose_n_envs_throughput_core(
    *,
    machine_id: str,
    proposed: dict[str, Any],
    gids: list[int],
    max_envs: int,
    per_candidate_s: float,
    min_iters: int,
    max_host_ram_pct: float,
    max_host_cpu_pct: float,
    host_wait_s: float,
    repo_root: Path,
    fleet_dir: Path,
    log: logging.Logger,
    make_probe_argv: Callable[[int], list[str]],
) -> tuple[int, dict[str, Any]]:
    """
    Probe ``n_envs`` from ``max(len(gids), baseline)`` .. ``max_envs`` (inclusive).

    Returns ``(winner_n_envs, report)``. On abort/skip, winner is baseline ``--n-envs``.
    """
    t_start = datetime.now(timezone.utc)
    fps_path = repo_root / "logs" / "fps_diag.jsonl"
    report: dict[str, Any] = {
        "schema": "throughput_tune.v1",
        "machine_id": machine_id,
        "started_at": t_start.isoformat(),
        "finished_at": None,
        "baseline_n_envs": None,
        "candidates": [],
        "winner_n_envs": None,
        "winner_median": None,
        "abort_reason": None,
    }

    try:
        baseline = _baseline_n_envs_from_proposed(proposed)
    except (KeyError, TypeError, ValueError) as exc:
        report["abort_reason"] = f"bad_proposed_n_envs:{exc}"
        report["winner_n_envs"] = None
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(fleet_dir / "throughput_tune.json", report)
        return 0, report

    report["baseline_n_envs"] = baseline
    floor_n = max(len(gids), baseline)
    ceiling_n = int(max_envs)

    if floor_n > ceiling_n:
        report["abort_reason"] = "floor_gt_ceiling"
        report["winner_n_envs"] = baseline
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(fleet_dir / "throughput_tune.json", report)
        return baseline, report

    if not _wait_host_headroom(
        max_host_ram_pct=max_host_ram_pct,
        max_host_cpu_pct=max_host_cpu_pct,
        host_wait_s=host_wait_s,
        log=log,
        phase="pre_sweep",
    ):
        report["abort_reason"] = "host_headroom_timeout_pre_sweep"
        report["winner_n_envs"] = baseline
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(fleet_dir / "throughput_tune.json", report)
        return baseline, report

    train_py = repo_root / "train.py"
    if not train_py.is_file():
        report["abort_reason"] = "missing_train_py"
        report["winner_n_envs"] = baseline
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(fleet_dir / "throughput_tune.json", report)
        return baseline, report

    best_n = baseline
    best_median: float | None = None

    for n in range(floor_n, ceiling_n + 1):
        if not _wait_host_headroom(
            max_host_ram_pct=max_host_ram_pct,
            max_host_cpu_pct=max_host_cpu_pct,
            host_wait_s=host_wait_s,
            log=log,
            phase=f"candidate_n_envs={n}",
        ):
            report["abort_reason"] = "host_headroom_timeout_mid_sweep"
            report["winner_n_envs"] = baseline
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            report["candidates"].append({"n_envs": n, "skipped": True, "reason": "host_busy"})
            _atomic_write_json(fleet_dir / "throughput_tune.json", report)
            return baseline, report

        base_argv = make_probe_argv(n)
        n_steps = _n_steps_from_argv(base_argv, proposed)
        if int(min_iters) > 0:
            target_iters = int(min_iters)
        else:
            target_iters = max(32768, 2 * n_steps * int(n))
        probe_argv = _argv_with_probe_iters(base_argv, target_iters)

        cmd = [sys.executable, str(train_py), *probe_argv]
        env = environ_for_train_subprocess()
        env["AWBW_MACHINE_ID"] = machine_id

        diag_off = fps_path.stat().st_size if fps_path.is_file() else 0
        popen_kw: dict[str, Any] = {
            "cwd": str(repo_root),
            "env": env,
        }
        if sys.platform == "win32":
            # Python 3.12+: ``BELOW_NORMAL_PRIORITY_CLASS`` (not ``CREATE_BELOW_NORMAL_*``).
            _bf = getattr(
                subprocess,
                "BELOW_NORMAL_PRIORITY_CLASS",
                getattr(subprocess, "CREATE_BELOW_NORMAL_PRIORITY_CLASS", 0x00004000),
            )
            popen_kw["creationflags"] = int(_bf)

        log.info(
            "Probe n_envs=%s iters=%s (n_steps=%s) timeout=%.1fs",
            n,
            target_iters,
            n_steps,
            per_candidate_s,
        )
        proc = subprocess.Popen(cmd, **popen_kw)
        _lower_probe_priority(proc)
        timed_out = False
        exit_code: int | None = None
        try:
            exit_code = proc.wait(timeout=float(per_candidate_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                exit_code = proc.wait(timeout=60)
            except (OSError, subprocess.SubprocessError):
                exit_code = None

        new_text = _read_diag_bytes_from_offset(fps_path, diag_off)
        totals = parse_fps_diag_lines(new_text)
        collects = parse_fps_diag_collect_lines(new_text) if not totals else []
        values = parse_fps_diag_throughput_values(new_text)
        stats = summarize_fps(values)
        median = stats.get("median")
        median_f = float(median) if median is not None else float("nan")

        row: dict[str, Any] = {
            "n_envs": n,
            "target_iters": target_iters,
            "train_exit_code": exit_code,
            "timed_out": timed_out,
            "n_samples_total": len(totals),
            "n_samples_collect": len(collects),
            "n_samples_scored": stats["n_samples"],
            "median": median,
            "summarize_fps": stats,
            "used_collect_fallback": bool(not totals and collects),
        }
        report["candidates"].append(row)

        if stats["n_samples"] > 0 and math.isfinite(median_f):
            if (
                best_median is None
                or median_f > best_median
                or (median_f == best_median and n < best_n)
            ):
                best_median = median_f
                best_n = n

    report["winner_n_envs"] = best_n
    report["winner_median"] = best_median
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    if best_median is None:
        report["abort_reason"] = report.get("abort_reason") or "no_fps_samples"
        report["winner_n_envs"] = baseline

    _atomic_write_json(fleet_dir / "throughput_tune.json", report)
    return int(report["winner_n_envs"]), report


def choose_n_envs_throughput(
    *,
    machine_id: str | None = None,
    proposed: dict[str, Any] | None = None,
    gids: list[int] | None = None,
    max_envs: int | None = None,
    per_candidate_s: float = 120.0,
    min_iters: int = 4,
    max_host_ram_pct: float = 90.0,
    max_host_cpu_pct: float = 90.0,
    host_wait_s: float = 45.0,
    repo_root: Path | None = None,
    fleet_dir: Path | None = None,
    log: logging.Logger | None = None,
    make_probe_argv: Callable[[int], list[str]] | None = None,
    baseline_n: int | None = None,
    max_n: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Preferred API: pass ``machine_id``, ``proposed``, ``gids``, ``max_envs``,
    ``fleet_dir``, ``log``, and ``make_probe_argv`` (omit ``baseline_n``).

    **Legacy** (``scripts/start_solo_training.py``): ``baseline_n=``, ``max_n=``,
    ``make_probe_argv``, timing/host kwargs, ``repo_root``; uses
    ``$AWBW_MACHINE_ID`` (or ``tune``) and ``<repo>/fleet/<id>/``.
    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]

    if baseline_n is not None:
        if max_n is None:
            raise TypeError("choose_n_envs_throughput legacy mode requires max_n= when baseline_n= is set")
        if make_probe_argv is None:
            raise TypeError("choose_n_envs_throughput legacy mode requires make_probe_argv=")
        sample = make_probe_argv(int(baseline_n))
        mid = (
            (str(machine_id).strip() if machine_id else None)
            or _machine_id_from_train_argv(sample)
            or str(os.environ.get("AWBW_MACHINE_ID") or "").strip()
            or "tune"
        )
        lg = log or logging.getLogger("throughput_tune")
        return choose_n_envs_throughput_core(
            machine_id=mid,
            proposed={"args": {"--n-envs": int(baseline_n)}},
            gids=[],
            max_envs=int(max_n),
            per_candidate_s=float(per_candidate_s),
            min_iters=int(min_iters),
            max_host_ram_pct=float(max_host_ram_pct),
            max_host_cpu_pct=float(max_host_cpu_pct),
            host_wait_s=float(host_wait_s),
            repo_root=root,
            fleet_dir=root / "fleet" / mid,
            log=lg,
            make_probe_argv=make_probe_argv,
        )

    if (
        machine_id is None
        or proposed is None
        or max_envs is None
        or fleet_dir is None
        or log is None
        or make_probe_argv is None
    ):
        raise TypeError(
            "choose_n_envs_throughput requires machine_id=, proposed=, max_envs=, "
            "fleet_dir=, log=, make_probe_argv= (or legacy baseline_n= / max_n=)"
        )
    return choose_n_envs_throughput_core(
        machine_id=machine_id,
        proposed=proposed,
        gids=list(gids) if gids is not None else [],
        max_envs=int(max_envs),
        per_candidate_s=float(per_candidate_s),
        min_iters=int(min_iters),
        max_host_ram_pct=float(max_host_ram_pct),
        max_host_cpu_pct=float(max_host_cpu_pct),
        host_wait_s=float(host_wait_s),
        repo_root=root,
        fleet_dir=fleet_dir,
        log=log,
        make_probe_argv=make_probe_argv,
    )


def _cli_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine-id", type=str, default=os.environ.get("AWBW_MACHINE_ID") or "tune")
    ap.add_argument(
        "--fleet-dir",
        type=Path,
        default=None,
        help="Fleet directory (default: <repo>/fleet/<machine-id>)",
    )
    ap.add_argument("--max-envs", type=int, default=12)
    ap.add_argument("--per-candidate-s", type=float, default=120.0)
    ap.add_argument("--min-iters", type=int, default=4)
    ap.add_argument("--max-host-ram-pct", type=float, default=90.0)
    ap.add_argument("--max-host-cpu-pct", type=float, default=90.0)
    ap.add_argument("--host-wait-s", type=float, default=45.0)
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    mid = str(args.machine_id).strip() or "tune"
    fleet_dir = args.fleet_dir or (repo_root / "fleet" / mid)
    proposed_path = fleet_dir / "proposed_args.json"
    if not proposed_path.is_file():
        print(f"missing {proposed_path}", file=sys.stderr)
        return 2
    proposed = json.loads(proposed_path.read_text(encoding="utf-8"))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("throughput_tune")

    def make_probe_argv(n_envs: int) -> list[str]:
        return [
            "--machine-id",
            mid,
            "--n-envs",
            str(n_envs),
            "--n-steps",
            str((proposed.get("args") or {}).get("--n-steps", 512)),
            "--batch-size",
            str((proposed.get("args") or {}).get("--batch-size", 256)),
        ]

    winner, rep = choose_n_envs_throughput_core(
        machine_id=mid,
        proposed=proposed,
        gids=[],
        max_envs=int(args.max_envs),
        per_candidate_s=float(args.per_candidate_s),
        min_iters=int(args.min_iters),
        max_host_ram_pct=float(args.max_host_ram_pct),
        max_host_cpu_pct=float(args.max_host_cpu_pct),
        host_wait_s=float(args.host_wait_s),
        repo_root=repo_root,
        fleet_dir=fleet_dir,
        log=log,
        make_probe_argv=make_probe_argv,
    )
    print(json.dumps({"winner_n_envs": winner, "report_path": str(fleet_dir / "throughput_tune.json")}, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
