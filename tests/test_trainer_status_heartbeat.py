"""Tests for ``SelfPlayTrainer._write_trainer_status`` heartbeat.

Phase 10/11 logging prereq (`trainer-status-heartbeat`): orchestrator's
stuck-worker detection reads ``fleet/<id>/status.json``; trainer must publish
one heartbeat per outer cycle without crashing on Samba hiccups.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.fleet_env import FleetConfig
from rl.self_play import SelfPlayTrainer


def _make_trainer(
    *,
    fleet_cfg: FleetConfig | None,
    checkpoint_dir: Path,
    n_envs: int = 4,
    save_every: int = 50_000,
) -> SelfPlayTrainer:
    """Construct a SelfPlayTrainer skeleton without running its heavy ``__init__``.

    The heartbeat helper only needs ``fleet_cfg``, ``n_envs``, ``save_every``,
    ``checkpoint_dir``, and the one-time-warn flag. Bypassing ``__init__``
    keeps the test off the map pool / checkpoint scan.
    """
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.fleet_cfg = fleet_cfg
    trainer.n_envs = n_envs
    trainer.save_every = save_every
    trainer.checkpoint_dir = checkpoint_dir
    trainer._heartbeat_machine_id_warned = False
    return trainer


def test_heartbeat_auxiliary_writes_under_shared_root(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    ck = tmp_path / "ckpt"
    ck.mkdir()
    cfg = FleetConfig(
        role="auxiliary",
        machine_id="pc-test-aux",
        shared_root=shared,
        repo_root=repo,
    )
    trainer = _make_trainer(fleet_cfg=cfg, checkpoint_dir=ck)

    trainer._write_trainer_status(steps_done=12_345, rate=27.5)

    status = shared / "fleet" / "pc-test-aux" / "status.json"
    assert status.is_file(), "auxiliary heartbeat must land under shared_root/fleet/<id>/"
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["role"] == "auxiliary"
    assert payload["machine_id"] == "pc-test-aux"
    assert payload["task"] == "train"
    assert payload["current_target"].endswith("latest.zip")
    assert payload["steps_done"] == 12_345
    assert payload["n_envs"] == 4
    assert payload["save_every"] == 50_000
    assert payload["checkpoint_dir"] == str(ck)
    assert payload["rate_steps_per_s"] == pytest.approx(27.5)
    assert "last_poll" in payload


def test_heartbeat_main_writes_under_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ck = repo / "checkpoints"
    ck.mkdir()
    cfg = FleetConfig(
        role="main",
        machine_id=None,
        shared_root=None,
        repo_root=repo,
    )
    trainer = _make_trainer(fleet_cfg=cfg, checkpoint_dir=ck)

    trainer._write_trainer_status(steps_done=0, rate=0.0)

    status = repo / "fleet" / "main" / "status.json"
    assert status.is_file(), "main heartbeat must land under repo_root/fleet/main/"
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["role"] == "main"
    assert payload["task"] == "train"


def test_heartbeat_auxiliary_without_machine_id_is_noop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    ck = tmp_path / "ckpt"
    ck.mkdir()
    cfg = FleetConfig(
        role="auxiliary",
        machine_id=None,
        shared_root=shared,
        repo_root=repo,
    )
    trainer = _make_trainer(fleet_cfg=cfg, checkpoint_dir=ck)

    trainer._write_trainer_status(steps_done=1, rate=1.0)
    trainer._write_trainer_status(steps_done=2, rate=2.0)  # second call: no second warn

    fleet_root = shared / "fleet"
    assert not fleet_root.exists() or not any(fleet_root.iterdir()), (
        "no fleet status file should be created when machine_id is missing"
    )

    out = capsys.readouterr().out
    assert out.count("heartbeat skipped") == 1, "warn must fire exactly once"


def test_heartbeat_no_fleet_cfg_is_noop(tmp_path: Path) -> None:
    ck = tmp_path / "ckpt"
    ck.mkdir()
    trainer = _make_trainer(fleet_cfg=None, checkpoint_dir=ck)
    trainer._write_trainer_status(steps_done=10, rate=1.0)
    assert not (tmp_path / "fleet").exists()


def test_heartbeat_swallows_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ck = tmp_path / "ckpt"
    ck.mkdir()
    cfg = FleetConfig(
        role="main",
        machine_id=None,
        shared_root=None,
        repo_root=repo,
    )
    trainer = _make_trainer(fleet_cfg=cfg, checkpoint_dir=ck)

    def _boom(*args, **kwargs):
        raise OSError("samba unavailable")

    import rl.fleet_env as fleet_env_mod

    monkeypatch.setattr(fleet_env_mod, "write_status_json", _boom)

    # Must not raise.
    trainer._write_trainer_status(steps_done=99, rate=3.14)

    out = capsys.readouterr().out
    assert "heartbeat write failed" in out
    assert not (repo / "fleet" / "main" / "status.json").exists()
