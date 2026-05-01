#!/usr/bin/env python3
"""Run all pairwise symmetric_checkpoint_eval jobs concurrently (one OS process per matchup).

By default each matchup runs games sequentially (--no-parallel) so many matchups can run in
parallel without multiplying multi-GB policy loads. Use --intra-parallel to run up to two games
per seating block inside each matchup (heavy RAM); then use --matchup-workers 1 unless you know
you have headroom.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_one(name: str, cmd: list[str], cwd: Path, phi_profile: str) -> tuple[str, int]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "AWBW_PHI_PROFILE": phi_profile}
    r = subprocess.run(cmd, cwd=cwd, env=env)
    return name, r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "logs" / "tournament_parallel",
        help="Directory for per-matchup JSON outputs",
    )
    ap.add_argument(
        "--matchup-workers",
        type=int,
        default=None,
        metavar="N",
        help="Max concurrent matchup processes (default: 1; raising parallelizes SMB/zip copies — use 3+ only if stable).",
    )
    ap.add_argument(
        "--intra-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run up to two games per seating block inside each matchup (high RAM). Default: false.",
    )
    ap.add_argument(
        "--phi-profile",
        choices=("balanced", "capture"),
        default="balanced",
        help="Φ α,β,κ preset for eval (passed to symmetric_checkpoint_eval). Default: balanced (capture skews κ up).",
    )
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    common = [
        str(ROOT / "scripts" / "symmetric_checkpoint_eval.py"),
        "--map-id",
        "123858",
        "--tier",
        "T3",
        "--co-p0",
        "1",
        "--co-p1",
        "1",
        "--games-first-seat",
        "4",
        "--games-second-seat",
        "3",
        "--seed",
        "0",
        "--max-env-steps",
        "0",
        "--max-days",
        "100",
        "--phi-profile",
        args.phi_profile,
    ]
    if args.intra_parallel:
        common.extend(["--parallel"])
    else:
        common.append("--no-parallel")
    models = [
        ("root_latest", r"Z:\checkpoints\latest.zip"),
        ("main_ck0043", r"Z:\checkpoints\checkpoint_0043.zip"),
        ("pool_pc_b_latest", r"Z:\checkpoints\pool\pc-b\latest.zip"),
    ]
    jobs: list[tuple[str, list[str]]] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            na, pa = models[i]
            nb, pb = models[j]
            jpath = out / f"sym_{na}_vs_{nb}.json"
            cmd = [
                sys.executable,
                *common,
                "--candidate",
                pa,
                "--baseline",
                pb,
                "--json-out",
                str(jpath),
            ]
            jobs.append((f"{na}_vs_{nb}", cmd))

    if args.matchup_workers is not None:
        max_w = args.matchup_workers
    else:
        # Default 1: concurrent matchups all copy zips to .tmp/; SMB roots (Z:) often error (WinError 1450) if hammered.
        max_w = 1
    max_w = max(1, min(len(jobs), max_w))
    print(
        f"[tournament] {len(jobs)} matchups, concurrent_matchup_processes={max_w}, "
        f"intra_parallel={args.intra_parallel}, phi_profile={args.phi_profile}"
    )
    failed: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=max_w) as ex:
        futs = {
            ex.submit(_run_one, name, cmd, ROOT, args.phi_profile): name for name, cmd in jobs
        }
        for fut in as_completed(futs):
            name, code = fut.result()
            print(f"[tournament] {name} exit={code}")
            if code != 0:
                failed.append((name, code))
    if failed:
        print("[tournament] failures:", failed, file=sys.stderr)
        return 1
    print("[tournament] all matchups completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
