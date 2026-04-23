"""
Phase 10e: fleet orchestrator.

Single-process driver that ticks every N minutes, reads fleet state
from the shared mount, and curates / evaluates / promotes / nudges
laggards. Default --dry-run: computes decisions, applies nothing.

This first cut is FILE-SYSTEM ONLY. Aux machines write heartbeats and
pool checkpoints to <shared>/fleet/ and <shared>/checkpoints/pool/;
this script just reads. SSH-driven probes are deferred to Phase 10f.

Audit trail: every tick appends one or more rows to
logs/fleet_orchestrator.jsonl (one row per decision class —
heartbeat-check, curate, eval, promote, reload).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.fleet_env import (  # noqa: E402
    prune_checkpoint_zip_curated,
    sorted_checkpoint_zip_paths,
    verdict_summary_from_symmetric_json,
)

DecKind = Literal["heartbeat_alert", "curate", "eval", "promote", "reload_request", "noop"]


@dataclass
class MachineState:
    machine_id: str
    status_path: Path
    status_json: Optional[dict[str, Any]]
    last_seen_seconds_ago: Optional[float]
    pool_dir: Path
    pool_latest_zip: Optional[Path]
    recent_verdict: Optional[dict[str, Any]]
    consecutive_laggard_cycles: int


@dataclass
class FleetState:
    machines: dict[str, MachineState]
    shared_checkpoints_dir: Path
    verdicts_dir: Path
    tick_started_at: float


@dataclass
class TickDecision:
    kind: DecKind
    machine_id: Optional[str]
    details: dict[str, Any]
    applied: bool
    reason: str


def _pool_latest_zip(pool_dir: Path) -> Optional[Path]:
    latest = pool_dir / "latest.zip"
    if latest.is_file():
        return latest
    paths = sorted_checkpoint_zip_paths(pool_dir)
    return paths[-1] if paths else None


def _read_json_path(p: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _status_last_seen_ago(data: dict[str, Any], now: float) -> Optional[float]:
    t = data.get("last_poll")
    if t is None:
        t = data.get("timestamp")
    if t is None:
        return None
    try:
        return max(0.0, now - float(t))
    except (TypeError, ValueError):
        return None


def _newest_verdict_under_machine(fleet: Path, machine_id: str) -> Optional[dict[str, Any]]:
    ev = fleet / machine_id / "eval"
    if not ev.is_dir():
        return None
    jsons = sorted(
        (p for p in ev.iterdir() if p.is_file() and p.suffix == ".json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in jsons:
        d = _read_json_path(p)
        if d is not None:
            return d
    return None


def _verdict_summary_for_machine(
    mid: str,
    fleet_machines: dict[str, MachineState],
    eval_new: dict[str, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if mid in eval_new:
        return eval_new[mid]
    ms = fleet_machines.get(mid)
    if ms is not None and ms.recent_verdict is not None:
        return verdict_summary_from_symmetric_json(ms.recent_verdict)
    return None


def _serialize_decision(
    d: TickDecision, *, tick_id: str, now_iso: str
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": d.kind,
        "machine_id": d.machine_id,
        "details": d.details,
        "applied": d.applied,
        "reason": d.reason,
        "tick_id": tick_id,
        "wall_time_iso": now_iso,
    }
    return out


class FleetOrchestrator:
    def __init__(
        self,
        *,
        shared_root: Path,
        pools: list[str],
        dry_run: bool,
        repo_root: Path,
        keep_newest: int,
        keep_top_winrate: int,
        keep_diversity: int,
        curator_min_age_minutes: float,
        map_id: int,
        tier: str,
        co_p0: int,
        co_p1: int,
        games_first_seat: int,
        games_second_seat: int,
        reload_margin: float,
        reload_consecutive: int,
        stuck_threshold_seconds: float,
        audit_log: Path,
        state_file: Path,
        eval_timeout_seconds: float,
        eval_seed: int = 0,
    ) -> None:
        self.shared_root = shared_root.resolve()
        self.pools = sorted(set(pools))
        self.dry_run = dry_run
        self.repo_root = repo_root.resolve()
        self.k_newest = int(keep_newest)
        self.m_top = int(keep_top_winrate)
        self.d_div = int(keep_diversity)
        self.curator_min_age_minutes = float(curator_min_age_minutes)
        self.map_id = int(map_id)
        self.tier = str(tier)
        self.co_p0 = int(co_p0)
        self.co_p1 = int(co_p1)
        self.games_first_seat = int(games_first_seat)
        self.games_second_seat = int(games_second_seat)
        self.reload_margin = float(reload_margin)
        self.reload_consecutive = int(reload_consecutive)
        self.stuck_threshold_seconds = float(stuck_threshold_seconds)
        self.audit_log = audit_log
        self.state_file = state_file
        self.eval_timeout_seconds = float(eval_timeout_seconds)
        self.eval_seed = int(eval_seed)
        self._laggard_cycles: dict[str, int] = self._load_laggard_state()

    def _load_laggard_state(self) -> dict[str, int]:
        p = self.state_file
        if not p.is_file():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        lc = raw.get("laggard_cycles")
        if not isinstance(lc, dict):
            return {}
        out: dict[str, int] = {}
        for k, v in lc.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def _save_laggard_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"laggard_cycles": self._laggard_cycles, "updated_at": time.time()},
            indent=2,
        )
        self.state_file.write_text(payload, encoding="utf-8")

    def _ensure_machine(
        self,
        machines: dict[str, MachineState],
        machine_id: str,
        now: float,
    ) -> None:
        if machine_id in machines:
            return
        st_path = self.shared_root / "fleet" / machine_id / "status.json"
        st = _read_json_path(st_path) if st_path.is_file() else None
        last_ago: Optional[float] = None
        if st is not None:
            last_ago = _status_last_seen_ago(st, now)
        pool_dir = self.shared_root / "checkpoints" / "pool" / machine_id
        rv = _newest_verdict_under_machine(self.shared_root / "fleet", machine_id)
        n_lag = int(self._laggard_cycles.get(machine_id, 0))
        machines[machine_id] = MachineState(
            machine_id=machine_id,
            status_path=st_path,
            status_json=st,
            last_seen_seconds_ago=last_ago,
            pool_dir=pool_dir,
            pool_latest_zip=_pool_latest_zip(pool_dir),
            recent_verdict=rv,
            consecutive_laggard_cycles=n_lag,
        )

    def read_fleet_state(self) -> FleetState:
        t0 = time.time()
        machines: dict[str, MachineState] = {}
        fleet = self.shared_root / "fleet"
        if fleet.is_dir():
            for sub in sorted(fleet.iterdir()):
                if not sub.is_dir():
                    continue
                mid = sub.name
                sp = sub / "status.json"
                st: Optional[dict[str, Any]] = None
                if sp.is_file():
                    st = _read_json_path(sp)
                last_ago: Optional[float] = None
                if st is not None:
                    last_ago = _status_last_seen_ago(st, t0)
                pool_dir = self.shared_root / "checkpoints" / "pool" / mid
                rv = _newest_verdict_under_machine(fleet, mid)
                n_lag = int(self._laggard_cycles.get(mid, 0))
                machines[mid] = MachineState(
                    machine_id=mid,
                    status_path=sp,
                    status_json=st,
                    last_seen_seconds_ago=last_ago,
                    pool_dir=pool_dir,
                    pool_latest_zip=_pool_latest_zip(pool_dir),
                    recent_verdict=rv,
                    consecutive_laggard_cycles=n_lag,
                )
        for mid in self.pools:
            self._ensure_machine(machines, mid, t0)
        for mid, ms in list(machines.items()):
            n_lag = int(self._laggard_cycles.get(mid, 0))
            if ms.consecutive_laggard_cycles != n_lag:
                ms.consecutive_laggard_cycles = n_lag
        return FleetState(
            machines=machines,
            shared_checkpoints_dir=self.shared_root / "checkpoints",
            verdicts_dir=fleet,
            tick_started_at=t0,
        )

    def check_heartbeats(self, state: FleetState) -> list[TickDecision]:
        out: list[TickDecision] = []
        now = time.time()
        for mid in self.pools:
            self._ensure_machine(state.machines, mid, now)
            ms = state.machines.get(mid)
            if ms is None:
                continue
            ago = ms.last_seen_seconds_ago
            is_stuck = ago is None or ago > self.stuck_threshold_seconds
            if is_stuck:
                desc = "missing" if ago is None else f"last seen {ago:.0f}s ago"
                out.append(
                    TickDecision(
                        kind="heartbeat_alert",
                        machine_id=mid,
                        details={"last_seen_seconds_ago": ago, "status_path": str(ms.status_path)},
                        applied=False,
                        reason=f"heartbeat: {mid} {desc} (threshold {self.stuck_threshold_seconds:g}s)",
                    )
                )
        return out

    def curate_pools(self, _state: FleetState) -> list[TickDecision]:
        out: list[TickDecision] = []
        vroot = self.shared_root / "fleet"
        root_ck = self.shared_root / "checkpoints"
        if root_ck.is_dir():
            d = prune_checkpoint_zip_curated(
                root_ck,
                k_newest=self.k_newest,
                m_top_winrate=self.m_top,
                d_diversity=self.d_div,
                verdicts_root=vroot,
                min_age_minutes=self.curator_min_age_minutes,
                dry_run=self.dry_run,
            )
            out.append(
                TickDecision(
                    kind="curate",
                    machine_id=None,
                    details={"path": str(root_ck), "curator_result": d},
                    applied=not self.dry_run
                    and bool(len(d.get("removed") or ())),
                    reason="curate shared checkpoints/ root",
                )
            )
        for mid in self.pools:
            pdir = self.shared_root / "checkpoints" / "pool" / mid
            if not pdir.is_dir():
                continue
            d = prune_checkpoint_zip_curated(
                pdir,
                k_newest=self.k_newest,
                m_top_winrate=self.m_top,
                d_diversity=self.d_div,
                verdicts_root=vroot,
                min_age_minutes=self.curator_min_age_minutes,
                dry_run=self.dry_run,
            )
            out.append(
                TickDecision(
                    kind="curate",
                    machine_id=mid,
                    details={"path": str(pdir), "curator_result": d},
                    applied=not self.dry_run
                    and bool(len(d.get("removed") or ())),
                    reason=f"curate pool {mid}",
                )
            )
        return out

    def _build_eval_cmd(
        self, candidate: Path, baseline: Path, json_out: Path
    ) -> list[str]:
        return [
            sys.executable,
            str(self.repo_root / "scripts" / "symmetric_checkpoint_eval.py"),
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--map-id",
            str(self.map_id),
            "--tier",
            self.tier,
            "--co-p0",
            str(self.co_p0),
            "--co-p1",
            str(self.co_p1),
            "--games-first-seat",
            str(self.games_first_seat),
            "--games-second-seat",
            str(self.games_second_seat),
            "--seed",
            str(self.eval_seed),
            "--deterministic",
            "--max-env-steps",
            "0",
            "--json-out",
            str(json_out),
        ]

    def run_symmetric_evals(
        self, state: FleetState
    ) -> tuple[list[TickDecision], dict[str, dict[str, Any]]]:
        """Returns (decisions, machine_id -> verdict summary) for this tick."""
        out: list[TickDecision] = []
        by_mid: dict[str, dict[str, Any]] = {}
        root_latest = self.shared_root / "checkpoints" / "latest.zip"
        fleet_dir = self.shared_root / "fleet"
        for mid in self.pools:
            ms = state.machines.get(mid)
            cand = ms.pool_latest_zip if ms is not None else _pool_latest_zip(
                self.shared_root / "checkpoints" / "pool" / mid
            )
            if cand is None or not root_latest.is_file():
                continue
            json_out = fleet_dir / mid / "eval" / f"{cand.stem}.json"
            cmd = self._build_eval_cmd(cand, root_latest, json_out)
            if self.dry_run:
                out.append(
                    TickDecision(
                        kind="eval",
                        machine_id=mid,
                        details={"would_run_cli": cmd, "json_out": str(json_out)},
                        applied=False,
                        reason=f"eval dry-run: {mid} vs root latest (symmetric)",
                    )
                )
                continue
            pr = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.eval_timeout_seconds,
            )
            if pr.returncode != 0:

                def _tail(s: str, n: int = 500) -> str:
                    return s[-n:] if len(s) > n else s

                out.append(
                    TickDecision(
                        kind="eval",
                        machine_id=mid,
                        details={
                            "failure": True,
                            "exit_code": pr.returncode,
                            "stderr_tail": _tail(pr.stderr or ""),
                            "stdout_tail": _tail(pr.stdout or ""),
                        },
                        applied=not self.dry_run,
                        reason=f"eval failed: {mid} (exit {pr.returncode})",
                    )
                )
                continue
            raw = _read_json_path(json_out)
            if raw is None:
                out.append(
                    TickDecision(
                        kind="eval",
                        machine_id=mid,
                        details={"failure": True, "error": "missing or bad json_out"},
                        applied=not self.dry_run,
                        reason=f"eval: {mid} produced no readable verdict at {json_out}",
                    )
                )
                continue
            summ = verdict_summary_from_symmetric_json(raw)
            summ["_candidate_path"] = str(cand.resolve())
            by_mid[mid] = summ
            out.append(
                TickDecision(
                    kind="eval",
                    machine_id=mid,
                    details={"json_out": str(json_out), "verdict_summary": summ, "command": cmd},
                    applied=not self.dry_run,
                    reason=f"eval ok: {mid}",
                )
            )
        return out, by_mid

    def decide_promotion(
        self,
        _state: FleetState,
        fresh_verdicts: dict[str, dict[str, Any]],
    ) -> list[TickDecision]:
        out: list[TickDecision] = []
        root_latest = self.shared_root / "checkpoints" / "latest.zip"
        thr_wr = 0.5 + self.reload_margin
        gthr = 2 * self.games_first_seat
        for mid in sorted(fresh_verdicts.keys()):
            s = fresh_verdicts[mid]
            wr = float(s.get("winrate", 0.0))
            gd = int(s.get("games_decided", 0))
            cand_path = s.get("_candidate_path")
            if not cand_path:
                continue
            cp = Path(str(cand_path))
            if wr <= thr_wr or gd < gthr or not root_latest.is_file() or not cp.is_file():
                continue
            out.append(
                TickDecision(
                    kind="promote",
                    machine_id=mid,
                    details={
                        "verdict": {k: v for k, v in s.items() if k != "_candidate_path"},
                        "candidate_path": str(cp),
                        "root_latest": str(root_latest),
                    },
                    applied=not self.dry_run,
                    reason=f"promote: {mid} winrate {wr:.3f} over threshold {thr_wr:.3f} with games_decided={gd}",
                )
            )
            if not self.dry_run:
                self._apply_promote(cp)
                break
        return out

    def _apply_promote(self, winner_path: Path) -> None:
        ck = self.shared_root / "checkpoints"
        latest = ck / "latest.zip"
        if not latest.is_file() or not winner_path.is_file():
            return
        promoted = ck / "promoted"
        promoted.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        trail = promoted / f"candidate_{ts}.zip"
        shutil.copy2(latest, trail)
        publishing = ck / "latest.zip.publishing"
        shutil.copy2(winner_path, publishing)
        os.replace(publishing, latest)

    def decide_reload(
        self,
        state: FleetState,
        eval_by_mid: dict[str, dict[str, Any]],
    ) -> list[TickDecision]:
        out: list[TickDecision] = []
        for mid in self.pools:
            vsum = _verdict_summary_for_machine(
                mid, state.machines, eval_by_mid
            )
            if vsum is None:
                continue
            wr = float(vsum.get("winrate", 0.0))
            if wr > self.reload_margin:
                self._laggard_cycles[mid] = 0
                continue
            n = int(self._laggard_cycles.get(mid, 0)) + 1
            if n >= self.reload_consecutive:
                out.append(
                    TickDecision(
                        kind="reload_request",
                        machine_id=mid,
                        details={
                            "laggard_cycles": n,
                            "winrate": wr,
                            "target_path": str(
                                (self.shared_root / "checkpoints" / "latest.zip").resolve()
                            ),
                        },
                        applied=not self.dry_run,
                        reason=(
                            f"reload: {mid} winrate {wr:.3f} <= margin {self.reload_margin:g} "
                            f"for {n} cycle(s)"
                        ),
                    )
                )
                if not self.dry_run:
                    self._write_reload_request(mid)
                self._laggard_cycles[mid] = 0
            else:
                self._laggard_cycles[mid] = n
        return out

    def _write_reload_request(self, machine_id: str) -> None:
        fleet = self.shared_root / "fleet" / machine_id
        fleet.mkdir(parents=True, exist_ok=True)
        target = (self.shared_root / "checkpoints" / "latest.zip").resolve()
        payload = {
            "target_zip": str(target),
            "reason": f"orchestrator: laggard ≤{int(self.reload_margin * 100)}% over {self.reload_consecutive} cycles",
            "issued_at": int(time.time()),
            "min_steps_done": 0,
        }
        final = fleet / "reload_request.json"
        tmp = fleet / "reload_request.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, final)

    def tick(self) -> list[TickDecision]:
        tick_id = str(uuid.uuid4())
        st = self.read_fleet_state()
        all_d: list[TickDecision] = []
        all_d.extend(self.check_heartbeats(st))
        all_d.extend(self.curate_pools(st))
        ev_decs, ev_map = self.run_symmetric_evals(st)
        all_d.extend(ev_decs)
        all_d.extend(self.decide_promotion(st, ev_map))
        all_d.extend(self.decide_reload(st, ev_map))
        if not all_d:
            all_d.append(
                TickDecision(
                    kind="noop",
                    machine_id=None,
                    details={},
                    applied=False,
                    reason="no actions this tick",
                )
            )
        self.append_audit_log(all_d, tick_id=tick_id)
        self._save_laggard_state()
        return all_d

    def append_audit_log(self, decisions: list[TickDecision], *, tick_id: str) -> None:
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.audit_log.open("a", encoding="utf-8") as f:
            for d in decisions:
                row = _serialize_decision(d, tick_id=tick_id, now_iso=now_iso)
                f.write(json.dumps(row) + "\n")

    def run_forever(self, tick_minutes: float) -> None:
        try:
            while True:
                self.tick()
                time.sleep(tick_minutes * 60.0)
        except KeyboardInterrupt:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shared-root", type=Path, required=True)
    ap.add_argument(
        "--pools",
        type=str,
        required=True,
        help="Comma-separated machine IDs to curate, e.g. pc-b,keras-aux",
    )
    ap.add_argument("--map-id", type=int, default=123858)
    ap.add_argument("--tier", type=str, default="T3")
    ap.add_argument("--co-p0", type=int, default=1)
    ap.add_argument("--co-p1", type=int, default=1)
    ap.add_argument("--games-first-seat", type=int, default=4)
    ap.add_argument("--games-second-seat", type=int, default=3)
    ap.add_argument("--keep-newest", type=int, default=8)
    ap.add_argument("--keep-top-winrate", type=int, default=12)
    ap.add_argument("--keep-diversity", type=int, default=4)
    ap.add_argument("--curator-min-age-minutes", type=float, default=5.0)
    ap.add_argument(
        "--reload-margin",
        type=float,
        default=0.25,
        help="Laggard threshold: winrate <= this triggers reload countdown",
    )
    ap.add_argument("--reload-consecutive", type=int, default=2)
    ap.add_argument("--stuck-threshold-seconds", type=float, default=1200.0)
    ap.add_argument("--eval-timeout-seconds", type=float, default=1800.0)
    ap.add_argument("--eval-seed", type=int, default=0, help="Seed passed to symmetric checkpoint eval")
    ap.add_argument("--tick-minutes", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="Run one tick and exit")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Compute decisions but apply nothing. DEFAULT.",
    )
    ap.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="Disable --dry-run and actually apply decisions.",
    )
    ap.add_argument(
        "--audit-log", type=Path, default=Path("logs") / "fleet_orchestrator.jsonl"
    )
    ap.add_argument(
        "--state-file", type=Path, default=Path("logs") / "fleet_orchestrator_state.json"
    )
    args = ap.parse_args()
    pools = [p.strip() for p in str(args.pools).split(",") if p.strip()]
    orch = FleetOrchestrator(
        shared_root=args.shared_root,
        pools=pools,
        dry_run=bool(args.dry_run),
        repo_root=REPO_ROOT,
        keep_newest=args.keep_newest,
        keep_top_winrate=args.keep_top_winrate,
        keep_diversity=args.keep_diversity,
        curator_min_age_minutes=args.curator_min_age_minutes,
        map_id=args.map_id,
        tier=args.tier,
        co_p0=args.co_p0,
        co_p1=args.co_p1,
        games_first_seat=args.games_first_seat,
        games_second_seat=args.games_second_seat,
        reload_margin=args.reload_margin,
        reload_consecutive=args.reload_consecutive,
        stuck_threshold_seconds=args.stuck_threshold_seconds,
        audit_log=args.audit_log,
        state_file=args.state_file,
        eval_timeout_seconds=args.eval_timeout_seconds,
        eval_seed=args.eval_seed,
    )
    if args.once:
        orch.tick()
    else:
        orch.run_forever(args.tick_minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
