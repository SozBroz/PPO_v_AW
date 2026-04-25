# -*- coding: utf-8 -*-
"""Phase 11 — auto-capture of per-machine ``mcts_off_baseline.json`` from the
fleet eval daemon (`scripts/fleet_eval_daemon.py::_maybe_capture_mcts_baseline`).

These tests pin the helper's behaviour, **not** the daemon poll loop. We never
spawn a real ``fleet_eval_daemon.py`` subprocess (it would hold the per-checkpoint
lock and walk the share). Instead we monkeypatch ``subprocess.run`` inside
``scripts.fleet_eval_daemon`` so the helper is exercised in isolation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts import fleet_eval_daemon as daemon
from tools.mcts_baseline import (
    DEFAULT_MAX_AGE_HOURS,
    MCTS_OFF_BASELINE_SCHEMA_VERSION,
    baseline_path,
)


def _fake_completed(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["python"], returncode=returncode, stdout="", stderr=stderr)


def _write_baseline_file(
    shared: Path,
    machine_id: str,
    *,
    captured_at: str,
    winrate: float = 0.42,
    games: int = 200,
) -> Path:
    """Write a synthetic mcts_off_baseline.json that ``read_baseline`` will accept."""
    out = baseline_path(machine_id, shared)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MCTS_OFF_BASELINE_SCHEMA_VERSION,
        "machine_id": machine_id,
        "captured_at": captured_at,
        "checkpoint_zip": "checkpoints/pool/" + machine_id + "/latest.zip",
        "checkpoint_zip_sha256": "a" * 64,
        "games_decided": games,
        "winrate_vs_pool": winrate,
        "mcts_mode": "off",
        "source": "tools/capture_mcts_baseline.py",
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Decision matrix
# ---------------------------------------------------------------------------


def test_capture_called_when_baseline_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # noqa: ANN001 - subprocess.run signature is broad
        calls.append(list(cmd))
        return _fake_completed(returncode=0)

    monkeypatch.setattr(daemon.subprocess, "run", _fake_run)

    status = daemon._maybe_capture_mcts_baseline(
        machine_id="pc-b",
        shared_root=shared,
        enabled=True,
        games=200,
        seed=0,
    )
    assert status == "captured"
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "tools.capture_mcts_baseline", "--machine-id"]
    assert cmd[4] == "pc-b"
    assert "--shared-root" in cmd
    assert str(shared) in cmd
    assert "--games" in cmd and "200" in cmd
    assert "--seed" in cmd and "0" in cmd


def test_capture_skipped_when_baseline_present_and_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_baseline_file(shared, "pc-b", captured_at=fresh)

    def _should_not_run(*_a, **_k):
        raise AssertionError("subprocess.run should not be invoked when baseline is fresh")

    monkeypatch.setattr(daemon.subprocess, "run", _should_not_run)

    status = daemon._maybe_capture_mcts_baseline(
        machine_id="pc-b",
        shared_root=shared,
        enabled=True,
    )
    assert status == "present"


def test_capture_called_when_baseline_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path
    # Two weeks old → strictly past the 168h boundary.
    old_dt = datetime.now(timezone.utc) - timedelta(hours=DEFAULT_MAX_AGE_HOURS + 24.0)
    _write_baseline_file(shared, "pc-b", captured_at=old_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return _fake_completed(returncode=0)

    monkeypatch.setattr(daemon.subprocess, "run", _fake_run)

    status = daemon._maybe_capture_mcts_baseline(
        machine_id="pc-b",
        shared_root=shared,
        enabled=True,
    )
    assert status == "stale-recaptured"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Failure modes — must NOT raise
# ---------------------------------------------------------------------------


def test_capture_failure_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        return _fake_completed(returncode=2, stderr="boom\nbad zip\n")

    monkeypatch.setattr(daemon.subprocess, "run", _fake_run)

    status = daemon._maybe_capture_mcts_baseline(
        machine_id="pc-b",
        shared_root=shared,
        enabled=True,
    )
    assert status == "failed"


def test_capture_timeout_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1.0))

    monkeypatch.setattr(daemon.subprocess, "run", _fake_run)

    status = daemon._maybe_capture_mcts_baseline(
        machine_id="pc-b",
        shared_root=shared,
        enabled=True,
        timeout_s=0.01,
    )
    assert status == "failed"


# ---------------------------------------------------------------------------
# Default-disabled gating
# ---------------------------------------------------------------------------


def test_capture_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path

    def _should_not_run(*_a, **_k):
        raise AssertionError("subprocess.run should not be invoked when feature is disabled")

    monkeypatch.setattr(daemon.subprocess, "run", _should_not_run)

    status = daemon._maybe_capture_mcts_baseline(
        machine_id="pc-b",
        shared_root=shared,
        enabled=False,
    )
    assert status == "skipped"


# ---------------------------------------------------------------------------
# Operator escape hatch — extra args via shlex
# ---------------------------------------------------------------------------


def test_extra_args_split_via_shlex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path
    captured: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        captured.append(list(cmd))
        return _fake_completed(returncode=0)

    monkeypatch.setattr(daemon.subprocess, "run", _fake_run)

    status = daemon._maybe_capture_mcts_baseline(
        machine_id="pc-b",
        shared_root=shared,
        enabled=True,
        extra_args="--map-id 123858 --tier T4",
    )
    assert status == "captured"
    assert len(captured) == 1
    cmd = captured[0]
    # All four extra tokens must be appended verbatim, in order.
    assert "--map-id" in cmd
    assert "123858" in cmd
    assert "--tier" in cmd
    assert "T4" in cmd
    # And they must come AFTER the canonical flags so capture_mcts_baseline.py
    # treats them as overrides, not as defaults preempted by the canonical block.
    canonical_seed_idx = cmd.index("--seed")
    assert cmd.index("--map-id") > canonical_seed_idx
