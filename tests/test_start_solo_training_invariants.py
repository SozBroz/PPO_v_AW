"""Invariants: proposed/applied args sync with fleet_orchestrator argv builder; zombie guard."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_sst():
    p = REPO / "scripts" / "start_solo_training.py"
    name = "start_solo_training_invariants"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_fleet_orch():
    sys.path.insert(0, str(REPO / "scripts"))
    import fleet_orchestrator as fo

    return fo


def _argv_body_tokens(argv: list[str]) -> list[str]:
    return sorted(argv[2:])


def test_argv_round_trip_matches_orchestrator_builder() -> None:
    m = _load_sst()
    fo = _load_fleet_orch()
    proposed = {
        "machine_id": "pc-b",
        "args": {"--n-envs": 2, "--n-steps": 256, "--batch-size": 128},
    }
    x = m._build_train_argv(
        proposed=proposed,
        machine_id="pc-b",
        train_extra=["--ent-coef", "0.02", "--mcts-mode", "off"],
    )
    d = m._argv_to_args_dict(x)
    y = fo.build_train_argv_from_proposed_args({"args": d}, repo_root=REPO)
    assert _argv_body_tokens(x) == _argv_body_tokens(y)
    assert d.get("--max-env-steps") == 10000
    assert d.get("--max-p1-microsteps") == 4000


def test_applied_sha256_matches_orchestrator_fn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _load_sst()
    fo = _load_fleet_orch()
    mid = "solo-inv"
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "start_solo_training",
            "--machine-id",
            mid,
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    fleet = tmp_path / "fleet" / mid
    fleet.mkdir(parents=True)
    proposed_doc = {
        "machine_id": mid,
        "proposed_at": 1.0,
        "based_on_probe_at": 2.0,
        "reasoning": "test",
        "args": {"--n-envs": 4, "--n-steps": 512, "--batch-size": 256},
    }
    proposed_path = fleet / "proposed_args.json"
    proposed_path.write_text(json.dumps(proposed_doc), encoding="utf-8")

    def fake_run(cmd: list, **kwargs: object):
        if "probe_machine_caps" in str(cmd):
            return MagicMock(returncode=0, stderr="", stdout="")
        if "propose_train_args" in str(cmd):
            return MagicMock(returncode=0, stderr="", stdout="")
        return MagicMock(returncode=0, stderr="", stdout="")

    train_mock = MagicMock()
    train_mock.pid = 90001
    train_mock.poll = MagicMock(return_value=None)
    orch_mock = MagicMock()
    orch_mock.poll = MagicMock(return_value=None)

    sleep_calls = {"n": 0}

    def sleep_side(_t: float) -> None:
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 1:
            train_mock.poll.return_value = 0

    monkeypatch.setattr(m.time, "sleep", sleep_side)

    with (
        patch.object(m.subprocess, "run", side_effect=fake_run),
        patch.object(
            m.subprocess,
            "Popen",
            side_effect=[train_mock, orch_mock],
        ),
    ):
        rc = m.main()
    assert rc == 1

    applied_path = fleet / "applied_args.json"
    applied = json.loads(applied_path.read_text(encoding="utf-8"))
    proposed_after = json.loads(proposed_path.read_text(encoding="utf-8"))
    orch_sha = fo.proposed_args_content_sha256(proposed_after)
    assert orch_sha is not None
    assert applied["args_content_sha256"] == orch_sha
    assert applied["args"] == proposed_after["args"]


def test_zombie_train_pid_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _load_sst()
    mid = "solo-z"
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    fleet = tmp_path / "fleet" / mid
    fleet.mkdir(parents=True)
    (fleet / "train.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "start_solo_training",
            "--machine-id",
            mid,
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    with patch.object(m, "_configure_logging"):
        rc = m.main()
    assert rc != 0
