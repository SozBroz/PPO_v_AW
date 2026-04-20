#!/usr/bin/env python3
"""
Auxiliary eval daemon: poll shared ``checkpoints/checkpoint_*.zip``, symmetric-eval vs
``promoted/best.zip`` (or ``--baseline-zip``), write ``fleet/<MACHINE_ID>/eval/<stem>.json``,
and optionally copy winners to ``promoted/candidate_<ts>.zip``.

Requires ``AWBW_MACHINE_ROLE=auxiliary`` (or ``--machine-role``), ``AWBW_MACHINE_ID``, and a
reachable ``AWBW_SHARED_ROOT`` (default ``Z:\\``).

Resilience: exponential backoff when the share errors; heartbeat ``fleet/<ID>/status.json``
each poll; per-checkpoint lock file avoids duplicate work on one box (see lock docstring).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl.fleet_env import (  # noqa: E402
    FleetConfig,
    REPO_ROOT,
    assert_aux_write_path,
    bootstrap_fleet_layout,
    load_machine_id,
    load_machine_role,
    load_shared_root_for_role,
    validate_fleet_at_startup,
    verdict_summary_from_symmetric_json,
    write_status_json,
)


def _try_lock(path: Path) -> object | None:
    """Exclusive create; return handle to keep until release, or None if busy."""
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, str(os.getpid()).encode("ascii", errors="replace"))
    return fd


def _release_lock(fd: object | None, path: Path) -> None:
    if fd is not None:
        try:
            os.close(fd)  # type: ignore[arg-type]
        except OSError:
            pass
    path.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=None, help="Repo root (default: parent of scripts/)")
    ap.add_argument("--machine-role", type=str, default=None)
    ap.add_argument("--shared-root", type=str, default=None)
    ap.add_argument("--poll-interval", type=float, default=120.0)
    ap.add_argument("--map-id", type=int, default=123858)
    ap.add_argument("--tier", type=str, default="T3")
    ap.add_argument("--co-p0", type=int, default=1)
    ap.add_argument("--co-p1", type=int, default=1)
    ap.add_argument("--games-first-seat", type=int, default=4)
    ap.add_argument("--games-second-seat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--min-margin", type=float, default=0.05, help="winrate must exceed 0.5 + margin")
    ap.add_argument("--baseline-zip", type=Path, default=None, help="Default: checkpoints/promoted/best.zip then latest.zip")
    ap.add_argument("--one-shot", action="store_true", help="Process at most one new checkpoint then exit")
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else REPO_ROOT
    role = load_machine_role(args.machine_role)
    mid = load_machine_id()
    if role == "auxiliary" and not mid:
        raise SystemExit("[fleet_eval_daemon] Set AWBW_MACHINE_ID (e.g. eval1)")
    shared = load_shared_root_for_role(role, args.shared_root)
    cfg = FleetConfig(role=role, machine_id=mid, shared_root=shared, repo_root=repo)
    validate_fleet_at_startup(cfg)
    if not cfg.is_auxiliary:
        print("[fleet_eval_daemon] Warning: role is not auxiliary; continuing for local smoke tests")

    ck_dir = repo / "checkpoints"
    promoted = ck_dir / "promoted"
    fleet_dir = repo / "fleet" / mid / "eval" if mid else repo / "fleet" / "_anon" / "eval"
    status_path = repo / "fleet" / mid / "status.json" if mid else repo / "fleet" / "_anon" / "status.json"
    bootstrap_fleet_layout(repo, machine_id=mid, role=role)

    backoff = 5.0
    max_backoff = 300.0

    while True:
        try:
            write_status_json(
                status_path,
                role=role,
                machine_id=mid,
                task="eval_daemon",
                current_target=None,
            )
            ckpts = sorted(ck_dir.glob("checkpoint_*.zip"), key=lambda p: p.stat().st_mtime)
            baseline = args.baseline_zip
            if baseline is None:
                b1 = promoted / "best.zip"
                b2 = ck_dir / "latest.zip"
                if b1.is_file():
                    baseline = b1
                elif b2.is_file():
                    baseline = b2
                else:
                    print("[fleet_eval_daemon] No baseline (promoted/best.zip or latest.zip); sleeping")
                    time.sleep(args.poll_interval)
                    continue

            if not baseline.is_file():
                print(f"[fleet_eval_daemon] Missing baseline {baseline}; sleeping")
                time.sleep(args.poll_interval)
                continue

            worked = False
            for ck in ckpts:
                stem = ck.stem
                verdict_path = fleet_dir / f"{stem}.json"
                if verdict_path.is_file():
                    continue
                lock_path = repo / "fleet" / mid / f"{stem}.lock" if mid else repo / "fleet" / "_anon" / f"{stem}.lock"
                fd = _try_lock(lock_path)
                if fd is None:
                    continue
                try:
                    tmp_json = fleet_dir / f".{stem}_sym.json"
                    sym = _ROOT / "scripts" / "symmetric_checkpoint_eval.py"
                    cmd = [
                        sys.executable,
                        str(sym),
                        "--candidate",
                        str(ck),
                        "--baseline",
                        str(baseline),
                        "--map-id",
                        str(args.map_id),
                        "--tier",
                        args.tier,
                        "--co-p0",
                        str(args.co_p0),
                        "--co-p1",
                        str(args.co_p1),
                        "--games-first-seat",
                        str(args.games_first_seat),
                        "--games-second-seat",
                        str(args.games_second_seat),
                        "--seed",
                        str(args.seed),
                        "--json-out",
                        str(tmp_json),
                    ]
                    subprocess.run(cmd, check=True, cwd=str(repo))
                    sym_data = json.loads(tmp_json.read_text(encoding="utf-8"))
                    tmp_json.unlink(missing_ok=True)
                    summary = verdict_summary_from_symmetric_json(sym_data)
                    summary["ckpt"] = ck.name
                    summary["opponent"] = str(baseline)
                    summary["timestamp"] = time.time()
                    summary["machine_id"] = mid
                    g = int(summary.get("games_decided", 0))
                    wr = float(summary.get("winrate", 0.0))
                    thr = g >= args.min_games and wr >= 0.5 + args.min_margin
                    summary["promotion_threshold_met"] = bool(thr)
                    summary["promoted_candidate_zip"] = None
                    if thr and cfg.shared_root and mid:
                        cand_rel = f"checkpoints/promoted/candidate_{int(time.time())}.zip"
                        cand_abs = assert_aux_write_path(repo / cand_rel, cfg.shared_root, mid)
                        shutil.copy2(ck, cand_abs)
                        summary["promoted_candidate_zip"] = cand_rel
                    fleet_dir.mkdir(parents=True, exist_ok=True)
                    verdict_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    print(f"[fleet_eval_daemon] verdict -> {verdict_path}")
                    worked = True
                except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                    print(f"[fleet_eval_daemon] error on {ck.name}: {exc}")
                finally:
                    _release_lock(fd, lock_path)

                if args.one_shot:
                    return 0
                break

            backoff = 5.0 if worked else min(max_backoff, backoff * 1.5)
            time.sleep(args.poll_interval if worked else backoff)
        except OSError as exc:
            print(f"[fleet_eval_daemon] share error: {exc}; backoff {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2.0)


if __name__ == "__main__":
    raise SystemExit(main())
