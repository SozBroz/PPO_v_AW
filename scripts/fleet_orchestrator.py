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
Curriculum stage transitions also append to logs/fleet_curriculum_changes.jsonl
(competence snapshot, JSON-safe floats).

If a tick raises before completing, a traceback is written to
``logs/fleet_orchestrator_last_crash.txt`` (same directory as the audit log)
and the process re-raises (so supervisors see a non-zero exit).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

_LOG = logging.getLogger(__name__)


def _json_safe_for_audit(obj: Any) -> Any:
    """Strip NaN/inf floats so :func:`json.dumps` can use ``allow_nan=False``."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe_for_audit(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe_for_audit(v) for v in obj]
    return obj


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.fleet_env import (  # noqa: E402
    prune_checkpoint_zip_curated,
    sorted_checkpoint_zip_paths,
    verdict_summary_from_symmetric_json,
)
from rl.live_games_resync import resync_live_games_for_train_cmd  # noqa: E402
from rl.train_launch_env import environ_for_train_subprocess  # noqa: E402
from rl.train_reconfig_log import append_train_reconfig_line  # noqa: E402

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

# Phase 11 Slice D: MCTS sim-budget escalator wiring. Direct package imports
# (matches tests/test_mcts_baseline.py and tests/test_mcts_escalator.py) so
# tests can patch ``tools.mcts_eval_summary.build_cycle_result`` and
# ``tools.mcts_baseline.read_baseline`` and have the orchestrator pick up the
# patched callable through the module reference.
import tools.mcts_baseline as _mcts_baseline  # noqa: E402
import tools.mcts_escalator as _mcts_escalator  # noqa: E402
import tools.mcts_eval_summary as _mcts_eval_summary  # noqa: E402

DecKind = Literal[
    "heartbeat_alert",
    "curate",
    "eval",
    "promote",
    "reload_request",
    "proposed_args",
    "restart_train",
    "train_zombie_heal",
    "train_restart_suppressed",
    "train_reconfig_applied",
    "mcts_health",
    "mcts_health_refresh",
    "mcts_gate_pending",
    "mcts_skip_host",
    "mcts_refuse_train_advisor",
    "mcts_baseline_missing",
    "mcts_escalator_no_data",
    "mcts_escalator_double",
    "mcts_escalator_hold",
    "mcts_escalator_drop_to_off",
    "mcts_escalator_stop_ask",
    "mcts_ev_unavailable",
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


# Soft reconfigure: PPO / vec-env geometry only (no policy obs-space change).
PPO_RECONFIG_ALLOWLIST: frozenset[str] = frozenset(
    {"--n-envs", "--n-steps", "--batch-size"}
)
OPERATOR_TRAIN_ARGS_OVERRIDE_NAME = "operator_train_args_override.json"

# ``refresh_proposed_train_args_documents`` rebuilds ``args`` from ``propose_from_probe`` +
# curriculum; these keys are copied from the *previous* ``proposed_args.json`` when present
# so ``start_solo_training`` injects (live PPO, async backend, rollout geometry…) survive orchestrator ticks.
#
# ``--n-envs`` / ``--max-env-steps``: after probe heuristic + curriculum merges, overlay from the
# last proposed (or fallback applied_args) unless ``operator_train_args_override.json`` pins the key
# — override wins via ``applied_overrides`` above.
_PRESERVE_TRAIN_ARGS_FROM_PREVIOUS_PROPOSED: frozenset[str] = frozenset(
    {
        "--n-envs",
        "--max-env-steps",
        "--live-games-id",
        "--live-snapshot-dir",
        "--live-learner-seats",
        "--training-backend",
    }
)

# Train subprocess defaults: Cython hot paths on unless ``train_launch_cmd.json``
# ``env`` sets these keys to ``"0"`` (explicit entries win in ``_merge_train_launch_env``).
DEFAULT_TRAIN_PERF_ENV: dict[str, str] = {
    "AWBW_CYTHON_BFS": "1",
    "AWBW_USE_CYTHON_ENCODER": "1",
    # Keep env-level Phi and engine-side capture shaping gate aligned before
    # ``engine.game`` imports in train workers.
    "AWBW_REWARD_SHAPING": "phi",
    # Small pressure against endless shaped-reward slogs; at 10k P0 steps this
    # is -0.4 plus the fixed truncation penalty below.
    "AWBW_TIME_COST": "0.00005",
    "AWBW_TRUNCATION_PENALTY": "0.25",
}


def _merge_train_launch_env(existing: Any) -> dict[str, str]:
    out = {**DEFAULT_TRAIN_PERF_ENV}
    if isinstance(existing, dict):
        for k, v in existing.items():
            out[str(k)] = str(v)
    return out


# Keys that curriculum ``args_overrides()`` or MCTS health gate emits.  Only a
# change in *these* keys triggers a ``maybe_restart_train_for_proposed_args`` hard
# restart; every other key in ``applied_args`` is preserved as-is.  This keeps
# ``--n-envs``, ``--n-steps``, ``--batch-size``, ``--training-backend``,
# live-game knobs, etc. frozen to whatever ``start_solo_training`` originally
# wrote — exactly one bootstrap, no probe churn.  ``--max-env-steps`` is
# significant so operator / proposed horizon changes respawn ``train.py``.
RESTART_SIGNIFICANT_KEYS: frozenset[str] = frozenset(
    {
        "--max-env-steps",
        # Curriculum stage keys
        "--learner-greedy-mix",
        "--egocentric-episode-prob",
        "--dual-gradient-self-play",
        "--dual-gradient-hist-prob",
        "--capture-move-gate",
        "--opening-book-prob",
        "--cold-opponent",
        "--curriculum-tag",
        "--map-id",
        "--co-p0",
        "--co-p1",
        "--tier",
        "--curriculum-broad-prob",
        "--opening-book",
        "--opening-book-seats",
        # MCTS health-gate keys
        "--mcts-mode",
        "--mcts-sims",
    }
)


def _restart_significant_args_hash(doc: dict[str, Any]) -> Optional[str]:
    """SHA-256 of only the non-None restart-significant keys from ``args``."""
    args = doc.get("args") if isinstance(doc.get("args"), dict) else None
    if args is None:
        return None
    significant = {
        k: args[k]
        for k in RESTART_SIGNIFICANT_KEYS
        if k in args and args[k] is not None
    }
    blob = json.dumps(significant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _restart_significant_args_match(
    proposed_doc: dict[str, Any], applied_doc: dict[str, Any]
) -> bool:
    """True when both sides agree on all non-None restart-significant keys."""
    p_args = (
        proposed_doc.get("args")
        if isinstance(proposed_doc.get("args"), dict)
        else {}
    )
    a_args = (
        applied_doc.get("args")
        if isinstance(applied_doc.get("args"), dict)
        else {}
    )
    # Treat None the same as absent: skip the key if either side doesn't assert a
    # specific value.  Only compare when both sides carry an explicit non-None
    # setting for that key.
    for k in RESTART_SIGNIFICANT_KEYS:
        pv = p_args.get(k)
        av = a_args.get(k)
        if pv is None or av is None:
            continue
        if pv != av:
            return False
    return True


def _merge_restart_args(
    proposed: dict[str, Any], applied: dict[str, Any]
) -> dict[str, Any]:
    """Build args dict for a curriculum-driven restart: start from ``applied`` args,
    overlay only restart-significant keys from ``proposed``."""
    pargs = proposed.get("args") if isinstance(proposed.get("args"), dict) else {}
    aargs = applied.get("args") if isinstance(applied.get("args"), dict) else {}
    merged: dict[str, Any] = {**aargs}
    for k in RESTART_SIGNIFICANT_KEYS:
        if k in pargs and pargs[k] is not None:
            merged[k] = pargs[k]
    return merged


def _arg_diff_keys(
    proposed_args: dict[str, Any], applied_args: dict[str, Any]
) -> set[str]:
    """Keys where proposed and applied differ (treats missing as unequal)."""
    pa = proposed_args or {}
    aa = applied_args or {}
    if not isinstance(pa, dict):
        pa = {}
    if not isinstance(aa, dict):
        aa = {}
    keys = set(pa) | set(aa)
    return {k for k in keys if pa.get(k) != aa.get(k)}


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

    def emit_optional(
        out: list[str], processed_keys: set[str], flag: str, default: Any
    ) -> None:
        """
        If ``args`` does not list *flag*, emit ``default`` (fleet bootstrap).
        If the value is JSON ``null`` (Python None), **omit** the flag so
        ``train.py`` uses its default (e.g. all GL Std maps, or random COs).
        If *default* is ``None`` and the flag is absent from ``args``, omit.
        """
        if flag in args_map and args_map[flag] is None:
            return
        val = g(flag, default)
        if val is None:
            return
        if isinstance(val, (list, tuple)):
            out.extend([flag, ",".join(str(int(x)) for x in val)])
        else:
            out.extend([flag, str(val)])
        processed_keys.add(flag)

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
    ]
    processed: set[str] = {
        "--iters",
        "--n-envs",
        "--n-steps",
        "--batch-size",
        "--save-every",
    }
    emit_optional(head, processed, "--map-id", 123858)
    emit_optional(head, processed, "--tier", "T3")
    # Pin tier without CO flags → train samples random COs from that tier each episode.
    co_p0_default: Any = 1
    co_p1_default: Any = 1
    if args_map.get("--tier") is not None:
        if "--co-p0" not in args_map:
            co_p0_default = None
        if "--co-p1" not in args_map:
            co_p1_default = None
    emit_optional(head, processed, "--co-p0", co_p0_default)
    emit_optional(head, processed, "--co-p1", co_p1_default)
    # Match train.py defaults; stage A/B (capture bootstrap) comes from curriculum
    # ``args_overrides`` after orchestrator merge — do not pre-inject draft 10g here.
    emit_optional(head, processed, "--cold-opponent", "random")
    emit_optional(head, processed, "--learner-greedy-mix", 0.0)
    emit_optional(head, processed, "--egocentric-episode-prob", 0.0)
    dg = g("--dual-gradient-self-play", None)
    if dg is True or dg == _curriculum_advisor.FLAG_PRESENT:
        head.append("--dual-gradient-self-play")
        processed.add("--dual-gradient-self-play")
    head.extend(
        [
            "--max-env-steps",
            str(int(g("--max-env-steps", 10000))),
            "--max-p1-microsteps",
            str(int(g("--max-p1-microsteps", 4000))),
        ]
    )
    processed.update(
        {
            "--max-env-steps",
            "--max-p1-microsteps",
        }
    )
    mid_cli = g("--machine-id", None)
    if mid_cli is not None and str(mid_cli).strip() != "":
        head.extend(["--machine-id", str(mid_cli).strip()])
        processed.add("--machine-id")

    cm = g("--capture-move-gate", None)
    if cm is True or cm == _curriculum_advisor.FLAG_PRESENT:
        head.append("--capture-move-gate")
        processed.add("--capture-move-gate")
    elif isinstance(cm, (int, float)) and not isinstance(cm, bool):
        fv = float(cm)
        if 0.0 < fv <= 1.0:
            head.extend(["--capture-move-gate", str(fv)])
            processed.add("--capture-move-gate")

    lr = g("--log-replay-frames", None)
    if lr is True or lr == _curriculum_advisor.FLAG_PRESENT:
        head.append("--log-replay-frames")
        processed.add("--log-replay-frames")

    fd = g("--fps-diag", None)
    if fd is True or fd == _curriculum_advisor.FLAG_PRESENT:
        head.append("--fps-diag")
        processed.add("--fps-diag")

    live_raw = g("--live-games-id", None)
    if live_raw is not None:
        processed.add("--live-games-id")
        if isinstance(live_raw, list):
            live_ids = [int(x) for x in live_raw]
        else:
            live_ids = [int(live_raw)]
        for gid in live_ids:
            head.extend(["--live-games-id", str(gid)])

    for key in sorted(args_map.keys()):
        if key in processed:
            continue
        val = args_map[key]
        if val is True or val == _curriculum_advisor.FLAG_PRESENT:
            head.append(key)
        elif val is False:
            continue
        elif val is None:
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


def _cmdline_mentions_train_py(cmdline: list[str]) -> bool:
    return any(part and "train.py" in part for part in cmdline)


def _cmdline_machine_id_match(cmdline: list[str], machine_id: str) -> bool:
    flat = " ".join(cmdline)
    if f"--machine-id {machine_id}" in flat or f"--machine-id={machine_id}" in flat:
        return True
    i = 0
    while i < len(cmdline):
        if cmdline[i] == "--machine-id" and i + 1 < len(cmdline):
            return cmdline[i + 1] == machine_id
        if cmdline[i].startswith("--machine-id="):
            return cmdline[i].split("=", 1)[-1] == machine_id
        i += 1
    return False


def _process_matches_fleet_train(
    proc: Any, machine_id: str, cmdline: list[str]
) -> bool:
    import psutil

    if not _cmdline_mentions_train_py(cmdline):
        return False
    if _cmdline_machine_id_match(cmdline, machine_id):
        return True
    try:
        env = proc.environ()
    except (psutil.Error, AttributeError):
        env = {}
    return env.get("AWBW_MACHINE_ID") == machine_id


def list_fleet_train_pids_for_machine(machine_id: str) -> list[int]:
    """PIDs of Python processes running ``train.py`` for this fleet ``machine_id``."""
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return []
    out: list[int] = []
    self_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        pid = proc.info.get("pid")
        if pid is None or int(pid) == self_pid:
            continue
        raw = proc.info.get("cmdline")
        if not raw:
            continue
        cmdline = list(raw)
        name = (proc.info.get("name") or "").lower()
        if not name.startswith("python"):
            continue
        try:
            if _process_matches_fleet_train(proc, machine_id, cmdline):
                out.append(int(pid))
        except (psutil.Error, TypeError, ValueError):
            continue
    return sorted(set(out))


def _cleanup_fleet_train_processes_for_machine(machine_id: str) -> list[int]:
    """Terminate every ``train.py`` cohort for *machine_id* (best-effort)."""
    killed: list[int] = []
    for pid in list_fleet_train_pids_for_machine(machine_id):
        try:
            _terminate_train_process_tree(pid)
            killed.append(pid)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("cleanup train pid %s for %s: %s", pid, machine_id, exc)
    return killed


def _train_pid_identity_matches_machine(pid: int, machine_id: str) -> bool:
    import psutil

    try:
        proc = psutil.Process(pid)
        raw = proc.cmdline()
    except (psutil.Error, OSError, TypeError, ValueError):
        return False
    if not raw:
        return False
    cmdline = list(raw)
    name = (proc.name() or "").lower()
    if not name.startswith("python"):
        return False
    return _process_matches_fleet_train(proc, machine_id, cmdline)


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
        curriculum_window_games: int = 200,
        curriculum_state_file_template: str = "fleet/{machine_id}/curriculum_state.json",
        reconfig_ack_timeout_s: float = 120.0,
        train_zombie_heal_cooldown_s: float = 120.0,
        train_bootstrap_grace_s: float = 0.0,
        host_machine_id: str = "pc-b",
        enable_mcts_here: bool = False,
        mcts_health_window: int = 200,
        mcts_health_refresh_every_ticks: int = 1,
        mcts_gate_required_consecutive: int = 2,
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
        self.reconfig_ack_timeout_s = float(reconfig_ack_timeout_s)
        self.train_zombie_heal_cooldown_s = float(train_zombie_heal_cooldown_s)
        self.train_bootstrap_grace_s = float(train_bootstrap_grace_s)
        self.host_machine_id = str(host_machine_id)
        self.enable_mcts_here = bool(enable_mcts_here)
        self.mcts_health_window = int(mcts_health_window)
        self.mcts_health_refresh_every_ticks = int(mcts_health_refresh_every_ticks)
        self.mcts_gate_required_consecutive = max(1, int(mcts_gate_required_consecutive))
        self._last_apply_at_by_machine: dict[str, float] = {}
        self._train_restart_times_by_machine: dict[str, list[float]] = {}
        self._circuit_open_until_by_machine: dict[str, float] = {}
        self._last_zombie_heal_at_by_machine: dict[str, float] = {}
        state_doc = self._load_state()
        self._laggard_cycles: dict[str, int] = self._coerce_int_dict(
            state_doc.get("laggard_cycles")
        )
        self._mcts_pass_streak_by_machine: dict[str, int] = self._coerce_int_dict(
            state_doc.get("mcts_pass_streak_by_machine")
        )
        self._mcts_health_last_refresh_tick_by_machine: dict[str, int] = (
            self._coerce_int_dict(
                state_doc.get("mcts_health_last_refresh_tick_by_machine")
            )
        )
        try:
            self._tick_counter: int = int(state_doc.get("tick_counter", 0))
        except (TypeError, ValueError):
            self._tick_counter = 0

    def _load_state(self) -> dict[str, Any]:
        p = self.state_file
        if not p.is_file():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _coerce_int_dict(raw: Any) -> dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, int] = {}
        for k, v in raw.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "laggard_cycles": self._laggard_cycles,
                "mcts_pass_streak_by_machine": self._mcts_pass_streak_by_machine,
                "mcts_health_last_refresh_tick_by_machine": (
                    self._mcts_health_last_refresh_tick_by_machine
                ),
                "tick_counter": self._tick_counter,
                "updated_at": time.time(),
            },
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

    def refresh_mcts_health_documents(self, _state: FleetState) -> list[TickDecision]:
        """
        For each pool machine, recompute :func:`tools.mcts_health.compute_health`
        from ``<shared>/logs/<mid>/game_log.jsonl`` and atomically rewrite
        ``<shared>/fleet/<mid>/mcts_health.json`` (Phase 11d Slice A).

        Skips a machine when fewer than ``mcts_health_refresh_every_ticks`` ticks
        have elapsed since its last refresh. Setting the cadence to ``<= 0``
        disables refresh entirely (operator runs ``tools/mcts_health`` manually).
        Exceptions never crash the tick: they surface as ``applied=False``
        ``mcts_health_refresh`` rows.
        """
        out: list[TickDecision] = []
        every = self.mcts_health_refresh_every_ticks
        if every <= 0:
            return out
        for mid in self.pools:
            last = self._mcts_health_last_refresh_tick_by_machine.get(mid)
            if last is not None and (self._tick_counter - int(last)) < every:
                continue
            logs_dir = self.shared_root / "logs" / mid
            fleet_dir = self.shared_root / "fleet" / mid
            try:
                verdict = _mcts_health.compute_health(
                    mid, logs_dir, window=self.mcts_health_window
                )
                dest = _mcts_health.write_health_json(verdict, fleet_dir)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("mcts_health refresh failed for %s: %s", mid, exc)
                out.append(
                    TickDecision(
                        kind="mcts_health_refresh",
                        machine_id=mid,
                        details={
                            "machine_id": mid,
                            "logs_dir": str(logs_dir.resolve()),
                            "error": str(exc),
                        },
                        applied=False,
                        reason=f"mcts_health refresh failed for {mid}: {exc}",
                    )
                )
                continue
            self._mcts_health_last_refresh_tick_by_machine[mid] = self._tick_counter
            out.append(
                TickDecision(
                    kind="mcts_health_refresh",
                    machine_id=mid,
                    details={
                        "machine_id": mid,
                        "path": str(dest.resolve()),
                        "window": self.mcts_health_window,
                        "tick_counter": self._tick_counter,
                    },
                    applied=True,
                    reason=f"mcts_health refresh wrote {dest.name} for {mid}",
                )
            )
        return out

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

            prev: Optional[_curriculum_advisor.CurriculumState] = None
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

            mh_v = read_mcts_health(mid, self.shared_root)
            mh_mode_lc = (
                str(mh_v.proposed_mcts_mode).strip().lower()
                if mh_v is not None
                else "off"
            )
            mh_passing = (
                mh_v is not None and mh_v.pass_overall and mh_mode_lc != "off"
            )
            if mh_passing:
                streak = int(self._mcts_pass_streak_by_machine.get(mid, 0)) + 1
            else:
                streak = 0
            self._mcts_pass_streak_by_machine[mid] = streak

            mcts_aux_decisions: list[TickDecision] = []
            is_host = mid == self.host_machine_id
            if mh_passing and mh_v is not None:
                if mh_mode_lc == "train_advisor":
                    mcts_aux_decisions.append(
                        TickDecision(
                            kind="mcts_refuse_train_advisor",
                            machine_id=mid,
                            details={
                                "machine_id": mid,
                                "proposed_mcts_mode": mh_v.proposed_mcts_mode,
                                "proposed_mcts_sims": int(mh_v.proposed_mcts_sims),
                            },
                            applied=False,
                            reason=(
                                "mcts gate refused train_advisor mode "
                                f"for {mid}"
                            ),
                        )
                    )
                elif is_host and not self.enable_mcts_here:
                    mcts_aux_decisions.append(
                        TickDecision(
                            kind="mcts_skip_host",
                            machine_id=mid,
                            details={
                                "machine_id": mid,
                                "reason": (
                                    "operator-only on host; rerun with "
                                    "--enable-mcts-here to enable"
                                ),
                            },
                            applied=False,
                            reason=(
                                f"mcts merge skipped on host {mid} "
                                "(operator-only; pass --enable-mcts-here to enable)"
                            ),
                        )
                    )
                elif streak < self.mcts_gate_required_consecutive:
                    mcts_aux_decisions.append(
                        TickDecision(
                            kind="mcts_gate_pending",
                            machine_id=mid,
                            details={
                                "machine_id": mid,
                                "streak": streak,
                                "required": self.mcts_gate_required_consecutive,
                            },
                            applied=False,
                            reason=(
                                f"mcts gate pending for {mid}: streak "
                                f"{streak}/{self.mcts_gate_required_consecutive}"
                            ),
                        )
                    )
                else:
                    merged_args["--mcts-mode"] = mh_v.proposed_mcts_mode
                    merged_args["--mcts-sims"] = int(mh_v.proposed_mcts_sims)
                    merged_mcts = True

            override_path = (
                self.shared_root / "fleet" / mid / OPERATOR_TRAIN_ARGS_OVERRIDE_NAME
            )
            override_doc = _read_json_path(override_path) if override_path.is_file() else None
            override_args = (
                (override_doc or {}).get("args")
                if isinstance(override_doc, dict)
                else None
            )
            applied_overrides: dict[str, Any] = {}
            if isinstance(override_args, dict):
                for k, ov in override_args.items():
                    if not isinstance(k, str) or not k.startswith("--"):
                        _LOG.warning(
                            "ignoring non-flag key in %s for %s: %r",
                            OPERATOR_TRAIN_ARGS_OVERRIDE_NAME,
                            mid,
                            k,
                        )
                        continue
                    merged_args[k] = ov
                    applied_overrides[k] = ov

            old_doc = read_proposed_args(mid, self.shared_root)
            pin_args: dict[str, Any] = {}
            if isinstance(old_doc, dict):
                opa = old_doc.get("args")
                if isinstance(opa, dict):
                    for k in _PRESERVE_TRAIN_ARGS_FROM_PREVIOUS_PROPOSED:
                        if k not in opa:
                            continue
                        v = opa[k]
                        if v is None:
                            continue
                        if k == "--live-games-id" and isinstance(v, list) and len(v) == 0:
                            continue
                        pin_args[k] = v
            # Fallback: a bad tick can strip pins from proposed before this code shipped;
            # applied_args often still holds the last trainer the operator actually ran.
            applied_pin_path = self.shared_root / "fleet" / mid / "applied_args.json"
            applied_pin = _read_json_path(applied_pin_path)
            if isinstance(applied_pin, dict):
                apa = applied_pin.get("args")
                if isinstance(apa, dict):
                    for k in _PRESERVE_TRAIN_ARGS_FROM_PREVIOUS_PROPOSED:
                        if k in pin_args:
                            continue
                        if k not in apa:
                            continue
                        v = apa[k]
                        if v is None:
                            continue
                        if k == "--live-games-id" and isinstance(v, list) and len(v) == 0:
                            continue
                        pin_args[k] = v
            for k, v in pin_args.items():
                if k in applied_overrides:
                    continue
                merged_args[k] = v

            reasoning_parts = [str(base_doc.get("reasoning") or "")]
            if self.curriculum_enabled and curriculum_reason:
                reasoning_parts.append(f"curriculum: {curriculum_reason}")
            if merged_mcts and mh_v is not None:
                reasoning_parts.append(
                    f"mcts: mode={mh_v.proposed_mcts_mode} sims={mh_v.proposed_mcts_sims}"
                )
            if applied_overrides:
                keys_sorted = ", ".join(sorted(applied_overrides))
                reasoning_parts.append(f"override: {keys_sorted}")
            reasoning = "; ".join(p for p in reasoning_parts if p)

            prev_auto = True
            if isinstance(old_doc, dict):
                prev_auto = bool(old_doc.get("auto_apply", True))

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
                if prev is not None:
                    prev_norm = _curriculum_advisor.normalize_curriculum_stage_name(
                        prev.current_stage_name
                    )
                    new_norm = _curriculum_advisor.normalize_curriculum_stage_name(
                        st_new.current_stage_name
                    )
                    if prev_norm != new_norm:
                        ch_path = (
                            self.repo_root / "logs" / "fleet_curriculum_changes.jsonl"
                        )
                        ch_path.parent.mkdir(parents=True, exist_ok=True)
                        row = {
                            "ts": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "machine_id": mid,
                            "from_stage": prev_norm,
                            "to_stage": new_norm,
                            "reason": curriculum_reason,
                            "metrics": _json_safe_for_audit(metrics_snap),
                        }
                        with ch_path.open("a", encoding="utf-8") as ch_f:
                            ch_f.write(
                                json.dumps(row, sort_keys=True, allow_nan=False)
                                + "\n"
                            )

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
                            "operator_overrides": applied_overrides or None,
                        },
                        applied=args_changed and not self.dry_run,
                        reason=f"curriculum tick for {mid}",
                    )
                )
            out.extend(mcts_aux_decisions)
        return out

    def _write_train_reconfig_request(
        self, mid: str, proposed: dict[str, Any], request_id: str
    ) -> Path:
        """Write ``fleet/<id>/train_reconfig_request.json`` for the trainer to pick up."""
        fleet_dir = self.shared_root / "fleet" / mid
        fleet_dir.mkdir(parents=True, exist_ok=True)
        args0 = proposed.get("args")
        pargs: dict[str, Any] = {}
        for k in PPO_RECONFIG_ALLOWLIST:
            if isinstance(args0, dict) and k in args0:
                pargs[k] = args0[k]
        payload: dict[str, Any] = {
            "request_id": request_id,
            "args": pargs,
            "reason": "orchestrator: proposed vs applied (PPO geometry allowlist)",
        }
        final = fleet_dir / "train_reconfig_request.json"
        tmp = fleet_dir / "train_reconfig_request.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, final)
        return final

    def _poll_train_reconfig_ack(
        self, mid: str, request_id: str, deadline: float
    ) -> str | None:
        """Return *applied* / *failed* if a matching ack appears before *deadline*."""
        fleet_dir = self.shared_root / "fleet" / mid
        seen_applied: set[Path] = set()
        seen_failed: set[Path] = set()
        while time.time() < deadline:
            for pat in (
                "train_reconfig_request.applied.*.json",
                "train_reconfig_request.failed.*.json",
            ):
                is_fail = "failed" in pat
                for p in sorted(fleet_dir.glob(pat)):
                    if p in seen_applied or p in seen_failed:
                        continue
                    body = _read_json_path(p)
                    if not isinstance(body, dict):
                        continue
                    if str(body.get("request_id", "")) != str(request_id):
                        continue
                    if is_fail:
                        seen_failed.add(p)
                        return "failed"
                    seen_applied.add(p)
                    return "applied"
            time.sleep(0.25)
        return None

    def _respawn_train_from_launch_file(self, launch_path: Path) -> Optional[int]:
        raw = _read_json_path(launch_path)
        if raw is None:
            return None
        cmd = raw.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            return None
        env_extra = _merge_train_launch_env(raw.get("env"))
        cwd_raw = raw.get("cwd")
        cwd = str(cwd_raw) if cwd_raw else str(self.repo_root)
        resync_live_games_for_train_cmd(self.repo_root, cmd, cwd=Path(cwd))
        env_full = {**environ_for_train_subprocess(), **env_extra}
        kw: dict[str, Any] = {"cwd": cwd, "env": env_full}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        proc = subprocess.Popen(cmd, **kw)  # noqa: S603
        return int(proc.pid)

    def maybe_restart_train_for_proposed_args(self, _state: FleetState) -> list[TickDecision]:
        """
        When **restart-significant** keys (``RESTART_SIGNIFICANT_KEYS``) in ``proposed_args.json``
        drift from ``applied_args.json``, optionally terminate and respawn ``train.py``.

        Only the keys defined in ``RESTART_SIGNIFICANT_KEYS`` are compared for the
        restart decision.  All other keys (``--n-envs``, ``--n-steps``,
        ``--batch-size``, ``--training-backend``, live-game knobs, etc.) are
        **frozen** to the last ``applied_args.json`` on respawn — they are not
        mutated by probe churn or curriculum ticks.  Horizon changes
        (``--max-env-steps``) are restart-significant so ``train.py`` picks up new argv.

        Gated by the orchestrator's ``--auto-apply`` (``self.auto_apply``) plus
        cooldown / circuit-breaker. The ``auto_apply`` field inside
        ``proposed_args.json`` is not consulted here (it is preserved for
        audit/refresh and operator visibility).
        """
        out: list[TickDecision] = []
        now = time.time()
        for mid in self.pools:
            proposed = read_proposed_args(mid, self.shared_root)
            if proposed is None:
                continue
            # Significant-key hash (see RESTART_SIGNIFICANT_KEYS): restart when curriculum,
            # MCTS knobs, horizon, etc. drift. Keys not in that set (--n-envs, live games, …)
            # stay pinned from the previous applied/trainer state across probe churn.
            prop_ch = _restart_significant_args_hash(proposed)
            if prop_ch is None:
                continue
            applied_path = (self.shared_root / "fleet" / mid / "applied_args.json").resolve()
            applied = _read_json_path(applied_path) if applied_path.is_file() else None
            if applied is None:
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"curriculum_hash": prop_ch},
                        applied=False,
                        reason=(
                            "applied_args.json missing; bootstrap must seed applied_args.json"
                        ),
                    )
                )
                continue
            # Full hash retained for audit; restart-significant-key comparison governs drift.
            # None in either side means "use default" — not a real disagreement.
            if _restart_significant_args_match(proposed, applied):
                continue
            full_prop_h = proposed_args_content_sha256(proposed)
            prev_ch = _restart_significant_args_hash(applied)
            if prev_ch is None:
                prev_ch = _restart_significant_args_hash({"args": applied.get("args")})
            prev_h = applied.get("args_content_sha256")
            if not isinstance(prev_h, str):
                prev_h = proposed_args_content_sha256({"args": applied.get("args")})

            # Alias for downstream references (suppressed decisions, applied doc,
            # lifecycle log): the full hash still appears in audit fields.
            prop_h = full_prop_h

            pid_path = self._resolve_train_sidecar_path(mid, self.train_pid_file_template)
            launch_path = self._resolve_train_sidecar_path(
                mid, self.train_launch_cmd_file_template
            )

            if not self.auto_apply:
                out.append(
                    TickDecision(
                        kind="train_restart_suppressed",
                        machine_id=mid,
                        details={
                            "proposed_hash": prop_h,
                            "previous_hash": prev_h,
                            "pid_path": str(pid_path),
                            "launch_path": str(launch_path),
                            "suppress_reason": "orchestrator_auto_apply_off",
                        },
                        applied=False,
                        reason="orchestrator --auto-apply not set",
                    )
                )
                continue

            if now < self._circuit_open_until_by_machine.get(mid, 0.0):
                out.append(
                    TickDecision(
                        kind="train_restart_suppressed",
                        machine_id=mid,
                        details={
                            "proposed_hash": prop_h,
                            "circuit_open_until": self._circuit_open_until_by_machine[mid],
                            "suppress_reason": "circuit_breaker",
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
                        kind="train_restart_suppressed",
                        machine_id=mid,
                        details={
                            "proposed_hash": prop_h,
                            "seconds_since_last_apply": now - last_apply,
                            "cooldown_s": self.apply_cooldown_s,
                            "suppress_reason": "cooldown",
                        },
                        applied=False,
                        reason="apply cooldown active",
                    )
                )
                continue

            grace = float(self.train_bootstrap_grace_s or 0.0)
            if grace > 0.0:
                try:
                    applied_age = now - float(applied.get("applied_at", 0.0))
                except (TypeError, ValueError):
                    applied_age = grace + 1.0
                if applied_age < grace:
                    p0 = (
                        proposed.get("args")
                        if isinstance(proposed.get("args"), dict)
                        else {}
                    )
                    a0 = (
                        applied.get("args")
                        if isinstance(applied.get("args"), dict)
                        else {}
                    )
                    dk0 = sorted(_arg_diff_keys(p0, a0))
                    out.append(
                        TickDecision(
                            kind="train_restart_suppressed",
                            machine_id=mid,
                            details={
                                "proposed_hash": prop_h,
                                "previous_hash": prev_h,
                                "suppress_reason": "bootstrap_grace",
                                "bootstrap_grace_s": grace,
                                "applied_age_s": applied_age,
                                "arg_diff_keys": dk0,
                            },
                            applied=False,
                            reason=(
                                f"bootstrap grace ({applied_age:.1f}s < {grace:g}s): "
                                f"deferred args-drift restart; diff={dk0}"
                            ),
                        )
                    )
                    _LOG.info(
                        "train_restart_suppressed bootstrap_grace machine_id=%s "
                        "applied_age_s=%.2f diff_keys=%s",
                        mid,
                        applied_age,
                        dk0,
                    )
                    continue

            pid: Optional[int] = None
            if pid_path.is_file():
                try:
                    pid = int(
                        pid_path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
                    )
                except (OSError, ValueError, IndexError):
                    _LOG.warning("train pid file unreadable for %s: %s", mid, pid_path)
                    pid = None
            else:
                _LOG.warning(
                    "train pid file missing for %s: %s (orchestrator will respawn if launch ok)",
                    mid,
                    pid_path,
                )

            if pid is not None and not _train_pid_process_alive(pid):
                _LOG.warning("train pid %s not running for %s", pid, mid)

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
                        kind="train_restart_suppressed",
                        machine_id=mid,
                        details={
                            "proposed_hash": prop_h,
                            "recent_restarts": len(hist),
                            "suppress_reason": "circuit_breaker",
                        },
                        applied=False,
                        reason="circuit breaker tripped (3 restarts in 30m)",
                    )
                )
                continue

            pargs = (
                proposed.get("args") if isinstance(proposed.get("args"), dict) else {}
            )
            aargs = applied.get("args") if isinstance(applied.get("args"), dict) else {}
            diff_k = _arg_diff_keys(pargs, aargs)
            attempted_soft = False
            if (
                diff_k
                and diff_k.issubset(PPO_RECONFIG_ALLOWLIST)
                and not self.dry_run
            ):
                attempted_soft = True
                request_id = str(int(time.time() * 1000))
                t_req = time.time()
                self._write_train_reconfig_request(mid, proposed, request_id)
                ack_deadline = t_req + self.reconfig_ack_timeout_s
                outcome = self._poll_train_reconfig_ack(mid, request_id, ack_deadline)
                if outcome == "applied":
                    applied_doc = {
                        **proposed,
                        "applied_at": time.time(),
                        "args_content_sha256": prop_h,
                    }
                    _atomic_write_json(applied_path, applied_doc)
                    self._last_apply_at_by_machine[mid] = time.time()
                    out.append(
                        TickDecision(
                            kind="train_reconfig_applied",
                            machine_id=mid,
                            details={
                                "proposed_hash": prop_h,
                                "request_id": request_id,
                                "applied_path": str(applied_path),
                            },
                            applied=True,
                            reason="in-process PPO geometry reconfig acknowledged",
                        )
                    )
                    append_train_reconfig_line(
                        self.shared_root,
                        {
                            "event": "soft_reconfig",
                            "machine_id": mid,
                            "request_id": request_id,
                            "outcome": "applied",
                            "soft_reconfig_orchestrator_wait_ms": int(
                                (time.time() - t_req) * 1000
                            ),
                            "old_args": aargs,
                            "new_args": pargs,
                        },
                    )
                    continue
                append_train_reconfig_line(
                    self.shared_root,
                    {
                        "event": "soft_reconfig",
                        "machine_id": mid,
                        "request_id": request_id,
                        "outcome": (outcome or "timeout"),
                    },
                )
                # fall through to hard restart
            if self.dry_run:
                out.append(
                    TickDecision(
                        kind="train_restart_suppressed",
                        machine_id=mid,
                        details={
                            "proposed_hash": prop_h,
                            "suppress_reason": "dry_run",
                            "would_soft_reconfig": bool(
                                diff_k and diff_k.issubset(PPO_RECONFIG_ALLOWLIST)
                            ),
                        },
                        applied=False,
                        reason="dry-run: no train restart or reconfig",
                    )
                )
                continue

            t_hard0 = time.time()
            _LOG.info(
                "train hard_restart begin machine_id=%s file_pid=%s arg_diff_keys=%s "
                "proposed_hash=%s previous_hash=%s",
                mid,
                pid,
                sorted(diff_k),
                prop_h,
                prev_h,
            )
            try:
                cleaned = _cleanup_fleet_train_processes_for_machine(mid)
                _LOG.info(
                    "train restart cleanup machine_id=%s file_pid=%s terminated=%s",
                    mid,
                    pid,
                    cleaned,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.error("failed to cleanup train processes for %s: %s", mid, exc)
                out.append(
                    TickDecision(
                        kind="restart_train",
                        machine_id=mid,
                        details={"pid": pid, "error": str(exc)},
                        applied=False,
                        reason="terminate train cohort failed",
                    )
                )
                continue

            raw_launch = _read_json_path(launch_path)
            launch_payload: dict[str, Any]
            if isinstance(raw_launch, dict):
                launch_payload = {**raw_launch}
            else:
                launch_payload = {}
            # Build restart doc from applied_args + proposed overlay; non-significant keys
            # (--n-envs, live games, …) preserve the last bootstrap geometry from applied.
            restart_args = _merge_restart_args(proposed, applied)
            restart_doc: dict[str, Any] = {
                **proposed,
                "args": restart_args,
            }
            launch_payload["cmd"] = build_train_argv_from_proposed_args(
                restart_doc, repo_root=self.repo_root
            )
            launch_payload["env"] = _merge_train_launch_env(launch_payload.get("env"))
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
                "args": restart_args,
                "applied_at": time.time(),
                "args_content_sha256": prop_h,
            }
            _atomic_write_json(applied_path, applied_doc)
            self._last_apply_at_by_machine[mid] = time.time()
            hist.append(time.time())
            self._train_restart_times_by_machine[mid] = hist

            t_hard1 = time.time()
            hrm = int((t_hard1 - t_hard0) * 1000)
            append_train_reconfig_line(
                self.shared_root,
                {
                    "event": "hard_restart",
                    "machine_id": mid,
                    "outcome": (
                        "hard_fallback"
                        if attempted_soft
                        and diff_k.issubset(PPO_RECONFIG_ALLOWLIST)
                        else "hard_restart"
                    ),
                    "hard_restart_ms": hrm,
                    "old_args": aargs,
                    "new_args": pargs,
                },
            )
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
                        "arg_diff_keys": sorted(diff_k),
                    },
                    applied=True,
                    reason="proposed_args content changed",
                )
            )
        return out

    def maybe_heal_stale_train(self, _state: FleetState) -> list[TickDecision]:
        """
        When ``proposed_args.json`` and ``applied_args.json`` already agree but ``train.py``
        is missing, dead, duplicated, or the pid file no longer matches the live cohort,
        terminate stray ``train.py`` processes for this machine and respawn one trainer
        from ``train_launch_cmd.json`` (cmd/env refreshed from current ``proposed``).

        Gated by ``auto_apply`` and ``train_zombie_heal_cooldown_s``. Skipped when
        ``dry_run`` (no subprocesses). Does not trip the hash-drift restart circuit breaker.
        """
        out: list[TickDecision] = []
        now = time.time()
        try:
            import psutil  # noqa: F401, PLC0415
        except ImportError:
            return out
        if self.dry_run:
            return out
        if not self.auto_apply:
            return out
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
                continue
            # For the heal path, heal only when restart-significant keys agree
            # (proposed == applied on the restart-significant set).  If
            # they differ, drift restart handles it (not zombie heal).
            # None in either side means "use default" — agreement check
            # uses _restart_significant_args_match for consistency.
            if not _restart_significant_args_match(proposed, applied):
                continue
            prev_h = applied.get("args_content_sha256")
            if not isinstance(prev_h, str):
                prev_h = proposed_args_content_sha256({"args": applied.get("args")})

            launch_path = self._resolve_train_sidecar_path(
                mid, self.train_launch_cmd_file_template
            )
            pid_path = self._resolve_train_sidecar_path(mid, self.train_pid_file_template)
            if not launch_path.is_file():
                continue

            grace = float(self.train_bootstrap_grace_s or 0.0)
            if grace > 0.0:
                try:
                    applied_age = now - float(applied.get("applied_at", 0.0))
                except (TypeError, ValueError):
                    applied_age = grace + 1.0
                if applied_age < grace:
                    _LOG.info(
                        "maybe_heal_stale_train skip bootstrap_grace machine_id=%s "
                        "applied_age_s=%.2f (trainer may still be starting)",
                        mid,
                        applied_age,
                    )
                    continue

            alive_trains = list_fleet_train_pids_for_machine(mid)
            file_pid: Optional[int] = None
            if pid_path.is_file():
                try:
                    file_pid = int(
                        pid_path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
                    )
                except (OSError, ValueError, IndexError):
                    file_pid = None

            healthy = (
                len(alive_trains) == 1
                and file_pid is not None
                and alive_trains[0] == file_pid
                and _train_pid_identity_matches_machine(file_pid, mid)
            )
            if healthy:
                continue

            _LOG.info(
                "train_zombie_heal triggered machine_id=%s file_pid=%s alive_trains=%s "
                "healthy=False",
                mid,
                file_pid,
                alive_trains,
            )

            last_heal = float(self._last_zombie_heal_at_by_machine.get(mid, 0.0))
            if now - last_heal < self.train_zombie_heal_cooldown_s:
                out.append(
                    TickDecision(
                        kind="train_restart_suppressed",
                        machine_id=mid,
                        details={
                            "suppress_reason": "zombie_heal_cooldown",
                            "seconds_since_last": now - last_heal,
                            "cooldown_s": self.train_zombie_heal_cooldown_s,
                        },
                        applied=False,
                        reason="train zombie heal cooldown active",
                    )
                )
                continue

            try:
                cleaned = _cleanup_fleet_train_processes_for_machine(mid)
            except Exception as exc:  # noqa: BLE001
                _LOG.error("zombie heal cleanup failed machine_id=%s: %s", mid, exc)
                out.append(
                    TickDecision(
                        kind="train_zombie_heal",
                        machine_id=mid,
                        details={"error": str(exc)},
                        applied=False,
                        reason="zombie heal cleanup failed",
                    )
                )
                continue

            raw_launch = _read_json_path(launch_path)
            launch_payload: dict[str, Any]
            if isinstance(raw_launch, dict):
                launch_payload = {**raw_launch}
            else:
                launch_payload = {}
            # Heal respawn: build args from applied + curriculum overlay
            # (same freeze semantics as drift restart).
            heal_args = _merge_restart_args(proposed, applied)
            heal_doc: dict[str, Any] = {**proposed, "args": heal_args}
            launch_payload["cmd"] = build_train_argv_from_proposed_args(
                heal_doc, repo_root=self.repo_root
            )
            launch_payload["env"] = _merge_train_launch_env(launch_payload.get("env"))
            if "cwd" not in launch_payload:
                launch_payload["cwd"] = str(self.repo_root)
            _atomic_write_json(launch_path, launch_payload)

            new_pid = self._respawn_train_from_launch_file(launch_path)
            if new_pid is None:
                out.append(
                    TickDecision(
                        kind="train_zombie_heal",
                        machine_id=mid,
                        details={},
                        applied=False,
                        reason="zombie heal respawn failed (bad train_launch_cmd.json)",
                    )
                )
                continue

            _atomic_write_text(pid_path, str(new_pid) + "\n")
            self._last_zombie_heal_at_by_machine[mid] = now
            _LOG.info(
                "train_zombie_heal machine_id=%s cleaned=%s new_pid=%s file_pid_was=%s alive_was=%s",
                mid,
                cleaned,
                new_pid,
                file_pid,
                alive_trains,
            )
            try:
                life = self.shared_root / "logs" / "orchestrator_train_lifecycle.log"
                life.parent.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with life.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"{ts} machine_id={mid} event=zombie_heal cleaned={cleaned!s} "
                        f"new_pid={new_pid} file_pid_was={file_pid!s}\n"
                    )
            except OSError as exc:  # pragma: no cover
                _LOG.warning("could not append orchestrator_train_lifecycle.log: %s", exc)

            out.append(
                TickDecision(
                    kind="train_zombie_heal",
                    machine_id=mid,
                    details={
                        "new_pid": new_pid,
                        "cleaned_pids": cleaned,
                        "file_pid_was": file_pid,
                        "alive_trains_was": alive_trains,
                    },
                    applied=True,
                    reason=(
                        "respawned train.py (hashes aligned; missing pid, dead pid, stale pid, "
                        "and/or duplicate cohort)"
                    ),
                )
            )
        return out

    def _mcts_changes_allowed_on(self, machine_id: str) -> bool:
        """Phase 11 Slice D: same host gate used by ``refresh_proposed_train_args_documents``.

        The orchestrator host (``self.host_machine_id``) is operator-only by
        default; auxiliary machines may always have MCTS args mutated by the
        escalator. ``--enable-mcts-here`` lifts the host restriction.
        """
        if str(machine_id) == str(self.host_machine_id):
            return bool(self.enable_mcts_here)
        return True

    def _mutate_proposed_args_for_mcts(
        self,
        machine_id: str,
        *,
        sims: Optional[int] = None,
        mode_off: bool = False,
    ) -> Optional[Path]:
        """Atomically rewrite ``fleet/<id>/proposed_args.json`` MCTS args.

        Used by :meth:`run_mcts_escalator` for ``DOUBLE`` and
        ``DROP_TO_OFF``. Preserves every other key in ``args`` and the
        rest of the document. Returns the path written, or ``None`` if
        no proposed file exists yet (escalator caller already gated on
        the presence of a non-``off`` ``--mcts-mode`` so this should
        not happen in practice).
        """
        out_path = self.shared_root / "fleet" / machine_id / "proposed_args.json"
        if not out_path.is_file():
            return None
        doc = _read_json_path(out_path)
        if not isinstance(doc, dict):
            return None
        args = dict(doc.get("args") or {})
        if not isinstance(args, dict):
            args = {}
        if mode_off:
            args["--mcts-mode"] = "off"
            args.pop("--mcts-sims", None)
        if sims is not None:
            args["--mcts-sims"] = int(sims)
        new_doc = {
            **doc,
            "args": args,
            "proposed_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        _atomic_write_json(out_path, new_doc)
        return out_path

    def _run_mcts_escalator_one(self, mid: str) -> list[TickDecision]:
        """Per-machine escalator step. Exceptions propagate to the wrapper."""
        out: list[TickDecision] = []
        proposed = read_proposed_args(mid, self.shared_root)
        args = proposed.get("args") if isinstance(proposed, dict) else None
        if not isinstance(args, dict):
            return out
        mode_lc = str(args.get("--mcts-mode", "")).strip().lower()
        if not mode_lc or mode_lc == "off":
            return out

        if mid == self.host_machine_id and not self.enable_mcts_here:
            out.append(
                TickDecision(
                    kind="mcts_skip_host",
                    machine_id=mid,
                    details={
                        "machine_id": mid,
                        "stage": "escalator",
                        "reason": (
                            "operator-only on host; rerun with "
                            "--enable-mcts-here to enable escalator"
                        ),
                    },
                    applied=False,
                    reason=(
                        f"mcts escalator skipped on host {mid} "
                        "(operator-only; pass --enable-mcts-here to enable)"
                    ),
                )
            )
            return out

        baseline = _mcts_baseline.read_baseline(mid, self.shared_root)
        if baseline is None or _mcts_baseline.is_baseline_stale(baseline):
            out.append(
                TickDecision(
                    kind="mcts_baseline_missing",
                    machine_id=mid,
                    details={
                        "machine_id": mid,
                        "baseline_present": baseline is not None,
                        "baseline_path": str(
                            _mcts_baseline.baseline_path(mid, self.shared_root)
                        ),
                    },
                    applied=False,
                    reason=(
                        f"run tools/capture_mcts_baseline.py --machine-id {mid} first"
                    ),
                )
            )
            return out

        cycle = _mcts_eval_summary.build_cycle_result(
            mid, self.shared_root, baseline
        )
        if cycle is None:
            out.append(
                TickDecision(
                    kind="mcts_escalator_no_data",
                    machine_id=mid,
                    details={
                        "machine_id": mid,
                        "reason": "no eval verdicts on disk yet",
                    },
                    applied=False,
                    reason=(
                        f"mcts escalator: no eval data for {mid} "
                        "(symmetric eval daemon hasn't produced verdicts yet)"
                    ),
                )
            )
            return out

        # Phase 11d EV scrape: distinguish "EV missing" (None) from "EV
        # measured at 0.0" (build_cycle_result clamps None -> 0.0). We
        # re-scrape via the same module attribute so a single test
        # monkeypatch on _mcts_eval_summary.latest_explained_variance
        # propagates to both call sites. Informational only — the
        # cycle still runs and the escalator gate will HOLD on the
        # min_explained_variance threshold.
        ev_window = _mcts_eval_summary.DEFAULT_EV_WINDOW_SECONDS
        ev_scraped = _mcts_eval_summary.latest_explained_variance(
            str(mid),
            self.shared_root,
            recent_window_seconds=ev_window,
        )
        if ev_scraped is None:
            out.append(
                TickDecision(
                    kind="mcts_ev_unavailable",
                    machine_id=mid,
                    details={
                        "machine_id": mid,
                        "scalar_tag": "train/explained_variance",
                        "window_seconds": float(ev_window),
                        "logs_dir": str(self.shared_root / "logs"),
                        "logs_dir_machine": str(
                            self.shared_root / "logs" / str(mid)
                        ),
                    },
                    applied=False,
                    reason=(
                        f"no recent train/explained_variance samples in "
                        f"{ev_window:.0f}s under {self.shared_root / 'logs'} "
                        "(escalator will HOLD on EV threshold)"
                    ),
                )
            )

        state_path = _mcts_escalator.default_state_path(mid, self.shared_root)
        log_path = _mcts_escalator.default_cycle_log_path(self.shared_root)
        # ``apply=True`` writes ``state_after`` AND appends one JSONL row to
        # ``logs/mcts_escalator.jsonl`` — the prompt's "Always: append_cycle_log".
        proposal = _mcts_escalator.compute_sims_proposal(
            state_path, cycle, log_path=log_path, apply=True
        )
        action = proposal.action
        new_sims = int(proposal.proposed_sims)
        current_sims = int(cycle.sims)
        can_mutate = (
            self.auto_apply
            and not self.dry_run
            and self._mcts_changes_allowed_on(mid)
        )

        base_details: dict[str, Any] = {
            "machine_id": mid,
            "current_sims": current_sims,
            "proposed_sims": new_sims,
            "winrate_vs_pool": float(cycle.winrate_vs_pool),
            "mcts_off_baseline": float(cycle.mcts_off_baseline),
            "games_decided": int(cycle.games_decided),
            "engine_desyncs_in_cycle": int(cycle.engine_desyncs_in_cycle),
            "explained_variance": float(cycle.explained_variance),
            "reason": proposal.reason,
        }

        if action == _mcts_escalator.EscalatorAction.DOUBLE:
            if can_mutate:
                self._mutate_proposed_args_for_mcts(mid, sims=new_sims)
                out.append(
                    TickDecision(
                        kind="mcts_escalator_double",
                        machine_id=mid,
                        details=base_details,
                        applied=True,
                        reason=(
                            "mcts escalator: doubled sims "
                            f"{current_sims}->{new_sims} for {mid}"
                        ),
                    )
                )
            else:
                out.append(
                    TickDecision(
                        kind="mcts_escalator_double",
                        machine_id=mid,
                        details=base_details,
                        applied=False,
                        reason=(
                            "mcts escalator: would double "
                            f"{current_sims}->{new_sims} for {mid}, "
                            "but auto_apply/host gate blocks mutation"
                        ),
                    )
                )
        elif action == _mcts_escalator.EscalatorAction.HOLD:
            out.append(
                TickDecision(
                    kind="mcts_escalator_hold",
                    machine_id=mid,
                    details=base_details,
                    applied=False,
                    reason=(
                        "mcts escalator: hold at sims="
                        f"{current_sims} for {mid} ({proposal.reason})"
                    ),
                )
            )
        elif action == _mcts_escalator.EscalatorAction.DROP_TO_OFF:
            # Operator-visible: an engine desync triggered the drop. Reset the
            # health-gate hysteresis streak so the gate must re-prove itself
            # over self.mcts_gate_required_consecutive consecutive ticks
            # before --mcts-mode can be merged back in.
            if can_mutate:
                self._mutate_proposed_args_for_mcts(mid, mode_off=True)
                self._mcts_pass_streak_by_machine[mid] = 0
                out.append(
                    TickDecision(
                        kind="mcts_escalator_drop_to_off",
                        machine_id=mid,
                        details=base_details,
                        applied=True,
                        reason=(
                            "mcts escalator: ALERT engine desync detected for "
                            f"{mid}; --mcts-mode -> off, hysteresis reset"
                        ),
                    )
                )
            else:
                out.append(
                    TickDecision(
                        kind="mcts_escalator_drop_to_off",
                        machine_id=mid,
                        details=base_details,
                        applied=False,
                        reason=(
                            "mcts escalator: ALERT engine desync detected for "
                            f"{mid} but auto_apply/host gate blocks mutation"
                        ),
                    )
                )
        elif action == _mcts_escalator.EscalatorAction.STOP_ASK_OPERATOR:
            out.append(
                TickDecision(
                    kind="mcts_escalator_stop_ask",
                    machine_id=mid,
                    details=base_details,
                    applied=False,
                    reason=(
                        "mcts escalator: stop+ask operator at sims="
                        f"{current_sims} for {mid}"
                    ),
                )
            )
        return out

    def run_mcts_escalator(self, _state: FleetState) -> list[TickDecision]:
        """Phase 11 Slice D: per-pool-machine sim-budget escalator step.

        Runs *after* ``refresh_proposed_train_args_documents`` (which
        merges --mcts-mode/--mcts-sims when the health gate passes) and
        *before* ``maybe_heal_stale_train`` / ``maybe_restart_train_for_proposed_args``
        so any DOUBLE / DROP_TO_OFF mutation flows through the same restart
        path the same tick.
        """
        out: list[TickDecision] = []
        for mid in self.pools:
            try:
                out.extend(self._run_mcts_escalator_one(mid))
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("mcts escalator failed for %s: %s", mid, exc)
                out.append(
                    TickDecision(
                        kind="mcts_escalator_no_data",
                        machine_id=mid,
                        details={
                            "machine_id": mid,
                            "error": str(exc),
                        },
                        applied=False,
                        reason=f"mcts escalator failure for {mid}: {exc}",
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
        self._tick_counter += 1
        st = self.read_fleet_state()
        all_d: list[TickDecision] = []
        all_d.extend(self.check_heartbeats(st))
        all_d.extend(self.refresh_mcts_health_documents(st))
        all_d.extend(self.refresh_proposed_train_args_documents(st))
        all_d.extend(self.run_mcts_escalator(st))
        all_d.extend(self.read_mcts_health_decisions())
        all_d.extend(self.read_fleet_diagnosis_decisions())
        all_d.extend(self.read_proposed_train_arg_docs())
        all_d.extend(self.maybe_heal_stale_train(st))
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
        self._save_state()
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
                try:
                    self.tick()
                except Exception:
                    p = self.audit_log.parent / "fleet_orchestrator_last_crash.txt"
                    try:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            + "\n"
                            + "".join(traceback.format_exception(*sys.exc_info())),
                            encoding="utf-8",
                        )
                    except OSError as wexc:  # pragma: no cover - best-effort
                        _LOG.warning("could not write %s: %s", p, wexc)
                    _LOG.exception("fleet_orchestrator tick failed")
                    raise
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
        "--reconfig-ack-timeout-s",
        type=float,
        default=120.0,
        help="Wall seconds to wait for train_reconfig_request ack before hard restart.",
    )
    ap.add_argument(
        "--train-zombie-heal-cooldown-s",
        type=float,
        default=120.0,
        help=(
            "Minimum seconds between automatic train.py respawns when proposed/applied "
            "hashes already match but the cohort is missing, dead, duplicated, or stale "
            "(default 120)."
        ),
    )
    ap.add_argument(
        "--train-bootstrap-grace-s",
        type=float,
        default=0.0,
        help=(
            "After applied_args.json is written, suppress hash-drift restarts and "
            "zombie-heal for this many seconds (0=off). Solo bootstrap uses ~180s to "
            "avoid killing train.py while the first orchestrator tick merges curriculum/MCTS."
        ),
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
        default=200,
        help="Rolling game window for curriculum metrics (default 200, §10g narrative)",
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
    ap.add_argument(
        "--host-machine-id",
        type=str,
        default="pc-b",
        help=(
            "Machine id treated as the operator host for the MCTS gate "
            "(skipped unless --enable-mcts-here)."
        ),
    )
    ap.add_argument(
        "--enable-mcts-here",
        action="store_true",
        default=False,
        help=(
            "Allow MCTS args to be merged into proposed_args.json on the host "
            "machine. Default: skipped (operator-only)."
        ),
    )
    ap.add_argument(
        "--mcts-health-window",
        type=int,
        default=200,
        help="Rolling-game window for tools/mcts_health compute_health (default 200).",
    )
    ap.add_argument(
        "--mcts-health-refresh-every-ticks",
        type=int,
        default=1,
        help=(
            "Recompute fleet/<id>/mcts_health.json every N orchestrator ticks "
            "(default 1)."
        ),
    )
    ap.add_argument(
        "--mcts-gate-required-consecutive",
        type=int,
        default=2,
        help=(
            "Require this many consecutive passing MCTS health verdicts before "
            "merging --mcts-mode/--mcts-sims into proposed_args.json (default 2)."
        ),
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
        reconfig_ack_timeout_s=float(args.reconfig_ack_timeout_s),
        train_zombie_heal_cooldown_s=float(args.train_zombie_heal_cooldown_s),
        train_bootstrap_grace_s=float(args.train_bootstrap_grace_s),
        host_machine_id=str(args.host_machine_id),
        enable_mcts_here=bool(args.enable_mcts_here),
        mcts_health_window=int(args.mcts_health_window),
        mcts_health_refresh_every_ticks=int(args.mcts_health_refresh_every_ticks),
        mcts_gate_required_consecutive=int(args.mcts_gate_required_consecutive),
    )
    if args.once:
        orch.tick()
    else:
        orch.run_forever(args.tick_minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
