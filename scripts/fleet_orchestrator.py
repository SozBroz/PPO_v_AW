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
heartbeat-check, mcts health, proposed_args, curate, eval, promote, reload).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

_LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.fleet_env import (  # noqa: E402
    prune_checkpoint_zip_curated,
    sorted_checkpoint_zip_paths,
    verdict_summary_from_symmetric_json,
)

# tools/ is not a regular package (no __init__.py); load the health gate by path.
_mh_name = "awbw_mcts_health"
_mh_spec = importlib.util.spec_from_file_location(
    _mh_name, REPO_ROOT / "tools" / "mcts_health.py"
)
_mcts_health = importlib.util.module_from_spec(_mh_spec)
# Required before exec: dataclasses look up ``sys.modules[cls.__module__]``.
sys.modules[_mh_name] = _mcts_health
assert _mh_spec.loader
_mh_spec.loader.exec_module(_mcts_health)

_fd_name = "awbw_fleet_diagnosis"
_fd_spec = importlib.util.spec_from_file_location(
    _fd_name, REPO_ROOT / "tools" / "fleet_diagnosis.py"
)
_fleet_diagnosis = importlib.util.module_from_spec(_fd_spec)
sys.modules[_fd_name] = _fleet_diagnosis
assert _fd_spec.loader
_fd_spec.loader.exec_module(_fleet_diagnosis)

_ca_name = "awbw_curriculum_advisor"
_ca_spec = importlib.util.spec_from_file_location(
    _ca_name, REPO_ROOT / "tools" / "curriculum_advisor.py"
)
_curriculum_advisor = importlib.util.module_from_spec(_ca_spec)
sys.modules[_ca_name] = _curriculum_advisor
assert _ca_spec.loader
_ca_spec.loader.exec_module(_curriculum_advisor)

_pt_name = "awbw_propose_train_args"
_pt_spec = importlib.util.spec_from_file_location(
    _pt_name, REPO_ROOT / "tools" / "propose_train_args.py"
)
_propose_train_args = importlib.util.module_from_spec(_pt_spec)
sys.modules[_pt_name] = _propose_train_args
assert _pt_spec.loader
_pt_spec.loader.exec_module(_propose_train_args)

DecKind = Literal[
    "heartbeat_alert",
    "curate",
    "eval",
    "promote",
    "reload_request",
    "proposed_args",
    "restart_train",
    "mcts_health",
    "fleet_diagnosis",
    "curriculum_proposal",
    "noop",
]


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


def read_proposed_args(machine_id: str, repo_root: Path) -> Optional[dict[str, Any]]:
    """
    Read ``fleet/<machine_id>/proposed_args.json`` under *repo_root* (typically
    ``--shared-root`` for this orchestrator — the tree that contains ``fleet/``).
    """
    p = Path(repo_root) / "fleet" / machine_id / "proposed_args.json"
    if not p.is_file():
        return None
    return _read_json_path(p)


def proposed_args_content_sha256(proposed_doc: dict[str, Any]) -> Optional[str]:
    """SHA-256 of the canonical JSON for the ``args`` field (sorted keys)."""
    args = proposed_doc.get("args")
    if not isinstance(args, dict):
        return None
    blob = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def proposed_document_body_sha256(proposed_doc: dict[str, Any]) -> str:
    """SHA-256 of the document minus ``proposed_at`` (wall clock excluded from drift checks)."""
    body = {k: v for k, v in proposed_doc.items() if k != "proposed_at"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_train_argv_from_proposed_args(
    proposed_doc: dict[str, Any], *, repo_root: Path
) -> list[str]:
    """
    Build a ``train.py`` argv from ``proposed_args.json``-style ``args`` map.
    Mirrors ``scripts/start_solo_training.py`` defaults, extended to honor every key in ``args``.
    """
    args_map = proposed_doc.get("args")
    if not isinstance(args_map, dict):
        args_map = {}

    def g(key: str, default: Any) -> Any:
        return args_map[key] if key in args_map else default

    n_envs = int(g("--n-envs", 4))
    n_steps = int(g("--n-steps", 512))
    batch_size = int(g("--batch-size", 256))
    save_every = int(g("--save-every", 50_000))
    iters = int(g("--iters", 1_000_000_000))

    head: list[str] = [
        sys.executable,
        str(Path(repo_root) / "train.py"),
        "--iters",
        str(iters),
        "--n-envs",
        str(n_envs),
        "--n-steps",
        str(n_steps),
        "--batch-size",
        str(batch_size),
        "--save-every",
        str(save_every),
        "--map-id",
        str(g("--map-id", 123858)),
        "--tier",
        str(g("--tier", "T3")),
        "--co-p0",
        str(g("--co-p0", 1)),
        "--co-p1",
        str(g("--co-p1", 1)),
        "--cold-opponent",
        str(g("--cold-opponent", "greedy_capture")),
        "--learner-greedy-mix",
        str(g("--learner-greedy-mix", 0.3)),
        "--max-env-steps",
        str(int(g("--max-env-steps", 8000))),
        "--max-p1-microsteps",
        str(int(g("--max-p1-microsteps", 4000))),
    ]
    processed: set[str] = {
        "--iters",
        "--n-envs",
        "--n-steps",
        "--batch-size",
        "--save-every",
        "--map-id",
        "--tier",
        "--co-p0",
        "--co-p1",
        "--cold-opponent",
        "--learner-greedy-mix",
        "--max-env-steps",
        "--max-p1-microsteps",
    }
    cm = g("--capture-move-gate", None)
    if cm is True or cm == _curriculum_advisor.FLAG_PRESENT:
        head.append("--capture-move-gate")
        processed.add("--capture-move-gate")

    for key in sorted(args_map.keys()):
        if key in processed:
            continue
        val = args_map[key]
        if val is True or val == _curriculum_advisor.FLAG_PRESENT:
            head.append(key)
        elif val is False:
            continue
        else:
            head.extend([key, str(val)])
    return head


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _train_pid_process_alive(pid: int) -> bool:
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return False
    try:
        return bool(psutil.pid_exists(pid) and psutil.Process(pid).is_running())
    except (psutil.Error, ValueError):
        return False


def _terminate_train_process_tree(pid: int, timeout_s: float = 30.0) -> None:
    import psutil

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = list(proc.children(recursive=True)) + [proc]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout_s)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def read_mcts_health(
    machine_id: str, shared_root: Path
) -> Optional[_mcts_health.MctsHealthVerdict]:  # type: ignore[valid-type, misc]
    """
    Read ``fleet/<id>/mcts_health.json`` (Phase 11d). Never auto-applied.

    If ``measured_at`` is more than 24h old, returns a downgraded verdict with
    ``proposed_mcts_mode="off"`` and ``reasoning="stale verdict"`` (operator
    must re-run :mod:`tools.mcts_health`).

    Returns ``None`` if the file is missing or not parseable.
    """
    p = Path(shared_root) / "fleet" / machine_id / "mcts_health.json"
    if not p.is_file():
        return None
    raw = _read_json_path(p)
    if raw is None:
        return None
    v = _mcts_health.parse_mcts_health_json(raw)
    if v is None:
        return None
    if _mcts_health.is_mcts_health_stale(v.measured_at):
        return _mcts_health.stale_mcts_off_verdict(v)
    return v


def read_fleet_diagnosis(
    machine_id: str,
    shared_root: Path,
    *,
    is_orchestrator_host: bool = False,
    game_logs_path: Path | None = None,
    repo_root: Path | None = None,
) -> _fleet_diagnosis.DiagnosisVerdict:  # type: ignore[valid-type, misc]
    """Read or compute the most recent diagnosis verdict for this machine.

    Audit-only this session: returns the verdict for logging into
    fleet_orchestrator.jsonl. Never feeds into proposed_args.json.
    """
    shared_root = Path(shared_root)
    mid = str(machine_id)
    gpath = game_logs_path
    if gpath is None:
        rr = Path(repo_root) if repo_root is not None else shared_root
        cand = rr / "logs" / mid / "game_log.jsonl"
        gpath = cand if cand.is_file() else rr / "logs" / "game_log.jsonl"
    fleet_d = shared_root / "fleet" / mid
    applied = fleet_d / "applied_args.json"
    curr = fleet_d / "curriculum_state.json"
    ivhist = fleet_d / "intervention_history.json"
    v = _fleet_diagnosis.compute_diagnosis(
        gpath,
        mid,
        is_orchestrator_host=is_orchestrator_host,
        applied_args_path=applied if applied.is_file() else None,
        curriculum_state_path=curr if curr.is_file() else None,
        intervention_history_path=ivhist if ivhist.is_file() else None,
    )
    dest = fleet_d / "diagnosis.json"
    _fleet_diagnosis.write_diagnosis(dest, v)
    return v


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
        auto_apply: bool = False,
        apply_cooldown_s: float = 600.0,
        train_pid_file_template: str = "fleet/{machine_id}/train.pid",
        train_launch_cmd_file_template: str = "fleet/{machine_id}/train_launch_cmd.json",
        curriculum_enabled: bool = True,
        curriculum_window_games: int = 100,
        curriculum_state_file_template: str = "fleet/{machine_id}/curriculum_state.json",
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
        self.auto_apply = bool(auto_apply)
        self.apply_cooldown_s = float(apply_cooldown_s)
        self.train_pid_file_template = str(train_pid_file_template)
        self.train_launch_cmd_file_template = str(train_launch_cmd_file_template)
        self.curriculum_enabled = bool(curriculum_enabled)
        self.curriculum_window_games = int(curriculum_window_games)
        self.curriculum_state_file_template = str(curriculum_state_file_template)
        self._last_apply_at_by_machine: dict[str, float] = {}
        self._train_restart_times_by_machine: dict[str, list[float]] = {}
        self._circuit_open_until_by_machine: dict[str, float] = {}
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

    def read_mcts_health_decisions(self) -> list[TickDecision]:
        """Surface ``fleet/<id>/mcts_health.json`` in the audit log (read-only, never apply)."""
        out: list[TickDecision] = []
        for mid in self.pools:
            v = read_mcts_health(mid, self.shared_root)
            if v is None:
                continue
            p = self.shared_root / "fleet" / mid / "mcts_health.json"
            out.append(
                TickDecision(
                    kind="mcts_health",
                    machine_id=mid,
                    details={
                        "verdict": _mcts_health.verdict_to_dict(v),
                        "path": str(p.resolve()),
                    },
                    applied=False,
                    reason=f"mcts health gate (read-only, not applied): {mid}",
                )
            )
        return out

    def read_fleet_diagnosis_decisions(self) -> list[TickDecision]:
        """Write ``fleet/<id>/diagnosis.json`` and surface verdict in the audit log (audit-only)."""
        out: list[TickDecision] = []
        for mid in self.pools:
            gpath = self._game_log_path_for_machine(mid)
            is_host = mid == "pc-b"
            v = read_fleet_diagnosis(
                mid,
                self.shared_root,
                is_orchestrator_host=is_host,
                game_logs_path=gpath,
                repo_root=self.repo_root,
            )
            dest = (self.shared_root / "fleet" / mid / "diagnosis.json").resolve()
            out.append(
                TickDecision(
                    kind="fleet_diagnosis",
                    machine_id=mid,
                    details={
                        "event": "fleet_diagnosis",
                        "state": v.state.value,
                        "verdict": _fleet_diagnosis.verdict_to_dict(v),
                        "path": str(dest),
                    },
                    applied=False,
                    reason=f"fleet diagnosis (audit-only): {mid} state={v.state.value}",
                )
            )
        return out

    def read_proposed_train_arg_docs(self) -> list[TickDecision]:
        """Surface fleet/<id>/proposed_args.json in the audit log (read-only, never apply)."""
        out: list[TickDecision] = []
        for mid in self.pools:
            data = read_proposed_args(mid, self.shared_root)
            if data is None:
                continue
            path = self.shared_root / "fleet" / mid / "proposed_args.json"
            out.append(
                TickDecision(
                    kind="proposed_args",
                    machine_id=mid,
                    details={"proposed": data, "path": str(path.resolve())},
                    applied=False,
                    reason=f"proposed train args (read-only, not applied): {mid}",
                )
            )
        return out

    def _resolve_train_sidecar_path(self, machine_id: str, template: str) -> Path:
        rel = template.format(machine_id=machine_id)
        p = Path(rel)
        if p.is_absolute():
            return p.resolve()
        return (self.shared_root / p).resolve()

    def _game_log_path_for_machine(self, machine_id: str) -> Path:
        per_m = self.repo_root / "logs" / machine_id / "game_log.jsonl"
        if per_m.is_file():
            return per_m
        return self.repo_root / "logs" / "game_log.jsonl"

    def refresh_proposed_train_args_documents(self, _state: FleetState) -> list[TickDecision]:
        """
        Merge probe-derived args with curriculum advisor + MCTS health, atomically
        updating ``fleet/<id>/proposed_args.json`` when the payload changes.
        """
        out: list[TickDecision] = []
        for mid in self.pools:
            probe_path = self.shared_root / "fleet" / mid / "probe.json"
            if not probe_path.is_file():
                continue
            probe_raw = _read_json_path(probe_path)
            if probe_raw is None:
                continue
            try:
                base_doc = _propose_train_args.propose_from_probe(probe_raw)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("propose_from_probe failed for %s: %s", mid, exc)
                continue
            merged_args: dict[str, Any] = dict(base_doc.get("args") or {})
            if not isinstance(merged_args, dict):
                merged_args = {}

            curriculum_reason = ""
            curriculum_stage = ""
            metrics_snap: dict[str, Any] = {}
            st_new: Optional[_curriculum_advisor.CurriculumState] = None
            st_path = self._resolve_train_sidecar_path(
                mid, self.curriculum_state_file_template
            )
            merged_mcts = False

            if self.curriculum_enabled:
                gpath = self._game_log_path_for_machine(mid)
                prev = _curriculum_advisor.read_state(st_path)
                prop, st_new = _curriculum_advisor.compute_proposal_stable(
                    gpath,
                    prev,
                    window_games=self.curriculum_window_games,
                    machine_id=mid,
                )
                merged_args.update(prop.args_overrides)
                curriculum_reason = prop.reason
                curriculum_stage = prop.stage_name
                metrics_snap = asdict(prop.metrics_snapshot)

            v = read_mcts_health(mid, self.shared_root)
            if (
                v is not None
                and v.pass_overall
                and str(v.proposed_mcts_mode).strip().lower() != "off"
            ):
                merged_args["--mcts-mode"] = v.proposed_mcts_mode
                merged_args["--mcts-sims"] = int(v.proposed_mcts_sims)
                merged_mcts = True

            reasoning_parts = [str(base_doc.get("reasoning") or "")]
            if self.curriculum_enabled and curriculum_reason:
                reasoning_parts.append(f"curriculum: {curriculum_reason}")
            if merged_mcts and v is not None:
                reasoning_parts.append(
                    f"mcts: mode={v.proposed_mcts_mode} sims={v.proposed_mcts_sims}"
                )
            reasoning = "; ".join(p for p in reasoning_parts if p)

            prev_auto = False
            old_doc = read_proposed_args(mid, self.shared_root)
            if isinstance(old_doc, dict):
                prev_auto = bool(old_doc.get("auto_apply"))

            new_doc: dict[str, Any] = {
                **base_doc,
                "args": merged_args,
                "reasoning": reasoning,
                "auto_apply": prev_auto,
            }
            if self.curriculum_enabled and curriculum_stage:
                # Stage only (metrics go to audit row; omitting rolling metrics avoids
                # rewriting proposed_args.json every tick when args are unchanged).
                new_doc["curriculum"] = {"stage": curriculum_stage}

            out_path = self.shared_root / "fleet" / mid / "proposed_args.json"
            old_fp = (
                proposed_document_body_sha256(old_doc)
                if isinstance(old_doc, dict)
                else None
            )
            new_fp = proposed_document_body_sha256(new_doc)
            args_changed = old_fp != new_fp

            if self.curriculum_enabled and st_new is not None and not self.dry_run:
                _curriculum_advisor.write_state(st_path, st_new)

            if args_changed and not self.dry_run:
                new_doc["proposed_at"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                _atomic_write_json(out_path, new_doc)

            if self.curriculum_enabled:
                out.append(
                    TickDecision(
                        kind="curriculum_proposal",
                        machine_id=mid,
                        details={
                            "event": "curriculum_proposal",
                            "stage": curriculum_stage or None,
                            "metrics": metrics_snap or None,
                            "reason": curriculum_reason or None,
                            "merged_mcts": merged_mcts,
                            "args_changed": args_changed,
                            "path": str(out_path.resolve()),
                        },
                        applied=args_changed and not self.dry_run,
                        reason=f"curriculum tick for {mid}",
                    )
                )
        return out

    def _respawn_train_from_launch_file(self, launch_path: Path) -> Optional[int]:
        raw = _read_json_path(launch_path)
        if raw is None:
            return None
        cmd = raw.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            return None
        env_extra = raw.get("env") or {}
        if not isinstance(env_extra, dict):
            env_extra = {}
        cwd_raw = raw.get("cwd")
        cwd = str(cwd_raw) if cwd_raw else str(self.repo_root)
        env_full = {**os.environ, **{str(k): str(v) for k, v in env_extra.items()}}
        kw: dict[str, Any] = {"cwd": cwd, "env": env_full}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        proc = subprocess.Popen(cmd, **kw)  # noqa: S603
        return int(proc.pid)

    def maybe_restart_train_for_proposed_args(self, _state: FleetState) -> list[TickDecision]:
        """
        When ``proposed_args.json`` ``args`` hash differs from ``applied_args.json``,
        optionally terminate and respawn ``train.py`` (Tier 1: machine-probe-driven only).
        """
        out: list[TickDecision] = []
        now = time.time()
        for mid in self.pools:
            proposed = read_proposed_args(mid, self.shared_root)
            if proposed is None:
                continue
            prop_h = proposed_args_content_sha256(proposed)
            if prop_h is None:
                continue
            applied_path = (self.shared_root / "fleet" / mid / "applied_args.json").resolve()
            applied = _read_json_path(applied_path) if applied_path.is_file() else None
            if applied is None:
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"proposed_hash": prop_h},
                        applied=False,
                        reason=(
                            "applied_args.json missing; bootstrap must seed applied_args.json"
                        ),
                    )
                )
                continue
            prev_h = applied.get("args_content_sha256")
            if not isinstance(prev_h, str):
                prev_h = proposed_args_content_sha256({"args": applied.get("args")})
            if prev_h == prop_h:
                continue

            pid_path = self._resolve_train_sidecar_path(mid, self.train_pid_file_template)
            launch_path = self._resolve_train_sidecar_path(
                mid, self.train_launch_cmd_file_template
            )

            if not self.auto_apply:
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={
                            "proposed_hash": prop_h,
                            "previous_hash": prev_h,
                            "pid_path": str(pid_path),
                            "launch_path": str(launch_path),
                        },
                        applied=False,
                        reason="auto-apply disabled",
                    )
                )
                continue

            if now < self._circuit_open_until_by_machine.get(mid, 0.0):
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={
                            "circuit_open_until": self._circuit_open_until_by_machine[mid],
                        },
                        applied=False,
                        reason="circuit breaker open (3 restarts in 30m)",
                    )
                )
                continue

            last_apply = float(self._last_apply_at_by_machine.get(mid, 0.0))
            try:
                last_apply = max(last_apply, float(applied.get("applied_at", 0.0)))
            except (TypeError, ValueError):
                pass
            if now - last_apply < self.apply_cooldown_s:
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={
                            "seconds_since_last_apply": now - last_apply,
                            "cooldown_s": self.apply_cooldown_s,
                        },
                        applied=False,
                        reason="apply cooldown active",
                    )
                )
                continue

            if not pid_path.is_file():
                _LOG.warning("train pid file missing for %s: %s", mid, pid_path)
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"pid_path": str(pid_path)},
                        applied=False,
                        reason="train pid file missing (orchestrator does not spawn train)",
                    )
                )
                continue

            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip().split()[0])
            except (OSError, ValueError, IndexError):
                _LOG.warning("train pid file unreadable for %s: %s", mid, pid_path)
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"pid_path": str(pid_path)},
                        applied=False,
                        reason="train pid file missing or unreadable",
                    )
                )
                continue

            if not _train_pid_process_alive(pid):
                _LOG.warning("train pid %s not running for %s", pid, mid)
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"pid": pid},
                        applied=False,
                        reason="train pid stale or process not running",
                    )
                )
                continue

            if not launch_path.is_file():
                _LOG.error("train launch cmd file missing for %s: %s", mid, launch_path)
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"launch_path": str(launch_path)},
                        applied=False,
                        reason="train_launch_cmd.json missing",
                    )
                )
                continue

            hist = self._train_restart_times_by_machine.setdefault(mid, [])
            cutoff = now - 1800.0
            hist = [t for t in hist if t >= cutoff]
            self._train_restart_times_by_machine[mid] = hist
            if len(hist) >= 3:
                self._circuit_open_until_by_machine[mid] = now + 3600.0
                _LOG.error(
                    "circuit breaker: %s had 3 train restarts in 30m; suppressing 60m",
                    mid,
                )
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"recent_restarts": len(hist)},
                        applied=False,
                        reason="circuit breaker tripped (3 restarts in 30m)",
                    )
                )
                continue

            try:
                _terminate_train_process_tree(pid)
            except Exception as exc:  # noqa: BLE001
                _LOG.error("failed to terminate train pid %s: %s", pid, exc)
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"pid": pid, "error": str(exc)},
                        applied=False,
                        reason="terminate train failed",
                    )
                )
                continue

            raw_launch = _read_json_path(launch_path)
            launch_payload: dict[str, Any]
            if isinstance(raw_launch, dict):
                launch_payload = {**raw_launch}
            else:
                launch_payload = {}
            launch_payload["cmd"] = build_train_argv_from_proposed_args(
                proposed, repo_root=self.repo_root
            )
            if "env" not in launch_payload:
                launch_payload["env"] = {}
            if "cwd" not in launch_payload:
                launch_payload["cwd"] = str(self.repo_root)
            _atomic_write_json(launch_path, launch_payload)

            new_pid = self._respawn_train_from_launch_file(launch_path)
            if new_pid is None:
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={},
                        applied=False,
                        reason="respawn failed (bad train_launch_cmd.json)",
                    )
                )
                continue

            _atomic_write_text(pid_path, str(new_pid) + "\n")
            applied_doc = {
                **proposed,
                "applied_at": time.time(),
                "args_content_sha256": prop_h,
            }
            _atomic_write_json(applied_path, applied_doc)
            self._last_apply_at_by_machine[mid] = time.time()
            hist.append(time.time())
            self._train_restart_times_by_machine[mid] = hist

            _LOG.info(
                "restart_train applied: machine_id=%s stopped_pid=%s new_pid=%s",
                mid,
                pid,
                new_pid,
            )
            try:
                life = self.shared_root / "logs" / "orchestrator_train_lifecycle.log"
                life.parent.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with life.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"{ts} machine_id={mid} stopped_pid={pid} new_pid={new_pid} "
                        f"proposed_hash={prop_h!s}\n"
                    )
            except OSError as exc:  # pragma: no cover - best-effort log
                _LOG.warning("could not append orchestrator_train_lifecycle.log: %s", exc)

            out.append(
                TickDecision(
                    kind="restart_train",
                    machine_id=mid,
                    details={
                        "previous_pid": pid,
                        "new_pid": new_pid,
                        "proposed_hash": prop_h,
                        "applied_path": str(applied_path),
                    },
                    applied=True,
                    reason="proposed_args content changed",
                )
            )
        return out

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
        all_d.extend(self.refresh_proposed_train_args_documents(st))
        all_d.extend(self.read_mcts_health_decisions())
        all_d.extend(self.read_fleet_diagnosis_decisions())
        all_d.extend(self.read_proposed_train_arg_docs())
        all_d.extend(self.maybe_restart_train_for_proposed_args(st))
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
        "--auto-apply",
        action="store_true",
        default=False,
        help=(
            "Allow train.py restarts when fleet/<id>/proposed_args.json drifts vs "
            "applied_args.json (uses train.pid + train_launch_cmd.json). "
            "Independent of --dry-run (curate/eval/promote still follow --dry-run)."
        ),
    )
    ap.add_argument(
        "--apply-cooldown-s",
        type=float,
        default=600.0,
        help="Minimum wall seconds between train restarts per machine (default 600).",
    )
    ap.add_argument(
        "--train-pid-file",
        type=str,
        default="fleet/{machine_id}/train.pid",
        help="Path template under --shared-root for train.py PID file",
    )
    ap.add_argument(
        "--train-launch-cmd-file",
        type=str,
        default="fleet/{machine_id}/train_launch_cmd.json",
        help="Path template under --shared-root for train launch JSON",
    )
    ap.set_defaults(curriculum_enabled=True)
    ap.add_argument(
        "--no-curriculum",
        dest="curriculum_enabled",
        action="store_false",
        help="Disable curriculum advisor merge into proposed_args",
    )
    ap.add_argument(
        "--curriculum-enabled",
        dest="curriculum_enabled",
        action="store_true",
        help="Enable curriculum advisor merge (default)",
    )
    ap.add_argument(
        "--curriculum-window-games",
        type=int,
        default=100,
        help="Rolling game window for curriculum metrics",
    )
    ap.add_argument(
        "--curriculum-state-file",
        type=str,
        default="fleet/{machine_id}/curriculum_state.json",
        help="Template under --shared-root for curriculum_state.json",
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
        auto_apply=bool(args.auto_apply),
        apply_cooldown_s=args.apply_cooldown_s,
        train_pid_file_template=args.train_pid_file,
        train_launch_cmd_file_template=args.train_launch_cmd_file,
        curriculum_enabled=bool(args.curriculum_enabled),
        curriculum_window_games=int(args.curriculum_window_games),
        curriculum_state_file_template=str(args.curriculum_state_file),
    )
    if args.once:
        orch.tick()
    else:
        orch.run_forever(args.tick_minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
