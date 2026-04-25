"""fleet_orchestrator Tier 1: proposed_args hash + train restart guardrails."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    name = "fleet_orchestrator_auto_apply"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


fo = _fo()
FleetOrchestrator = fo.FleetOrchestrator
proposed_args_content_sha256 = fo.proposed_args_content_sha256


def _orch(tmp_path: Path, *, pools: list[str], **kw: object) -> object:
    defaults: dict = {
        "shared_root": tmp_path,
        "pools": pools,
        "dry_run": True,
        "repo_root": REPO,
        "keep_newest": 8,
        "keep_top_winrate": 12,
        "keep_diversity": 4,
        "curator_min_age_minutes": 5.0,
        "map_id": 123858,
        "tier": "T3",
        "co_p0": 1,
        "co_p1": 1,
        "games_first_seat": 4,
        "games_second_seat": 3,
        "reload_margin": 0.25,
        "reload_consecutive": 2,
        "stuck_threshold_seconds": 1200.0,
        "audit_log": tmp_path / "audit.jsonl",
        "state_file": tmp_path / "state.json",
        "eval_timeout_seconds": 1800.0,
        "eval_seed": 0,
        "auto_apply": False,
        "apply_cooldown_s": 600.0,
        "train_pid_file_template": "fleet/{machine_id}/train.pid",
        "train_launch_cmd_file_template": "fleet/{machine_id}/train_launch_cmd.json",
        "reconfig_ack_timeout_s": 120.0,
        "mcts_health_refresh_every_ticks": 0,
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _write_proposed(tmp: Path, mid: str, *, n_envs: int = 4) -> None:
    d = tmp / "fleet" / mid
    d.mkdir(parents=True, exist_ok=True)
    doc = {
        "machine_id": mid,
        "args": {"--n-envs": n_envs, "--n-steps": 512, "--batch-size": 256},
    }
    (d / "proposed_args.json").write_text(json.dumps(doc) + "\n", encoding="utf-8")


def _write_applied(tmp: Path, mid: str, doc: dict) -> None:
    d = tmp / "fleet" / mid
    d.mkdir(parents=True, exist_ok=True)
    h = proposed_args_content_sha256(doc)
    payload = {**doc, "applied_at": 1_000_000.0, "args_content_sha256": h}
    (d / "applied_args.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_hash_equal_for_same_args_different_key_order() -> None:
    h1 = proposed_args_content_sha256(
        {"args": {"--n-envs": 4, "--batch-size": 256, "--n-steps": 512}}
    )
    h2 = proposed_args_content_sha256(
        {"args": {"--batch-size": 256, "--n-steps": 512, "--n-envs": 4}}
    )
    assert h1 == h2
    assert h1 is not None


def test_hash_differs_when_args_change() -> None:
    h1 = proposed_args_content_sha256(
        {"args": {"--n-envs": 4, "--n-steps": 512, "--batch-size": 256}}
    )
    h2 = proposed_args_content_sha256(
        {"args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256}}
    )
    assert h1 != h2


def test_auto_apply_false_logs_restart_not_applied(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1", n_envs=4)
    _write_applied(
        tmp_path,
        "m1",
        {"machine_id": "m1", "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256}},
    )
    o = _orch(tmp_path, pools=["m1"], auto_apply=False)
    st = o.read_fleet_state()
    decs = o.maybe_restart_train_for_proposed_args(st)
    assert len(decs) == 1
    assert decs[0].kind == "train_restart_suppressed"
    assert decs[0].applied is False
    assert decs[0].details.get("suppress_reason") == "orchestrator_auto_apply_off"


def test_proposed_file_auto_apply_false_does_not_block_orchestrator_auto_apply(
    tmp_path: Path,
) -> None:
    """Orchestrator --auto-apply is the only gate; proposed JSON auto_apply is ignored here."""
    d = tmp_path / "fleet" / "m1"
    d.mkdir(parents=True)
    doc = {
        "machine_id": "m1",
        "args": {"--n-envs": 4, "--n-steps": 512, "--batch-size": 256},
        "auto_apply": False,
    }
    (d / "proposed_args.json").write_text(json.dumps(doc) + "\n", encoding="utf-8")
    applied_doc = {**doc, "args": {"--n-envs": 1, "--n-steps": 512, "--batch-size": 256}}
    ah = proposed_args_content_sha256(applied_doc)
    (d / "applied_args.json").write_text(
        json.dumps(
            {**applied_doc, "applied_at": 1.0, "args_content_sha256": ah}
        )
        + "\n",
        encoding="utf-8",
    )
    o = _orch(tmp_path, pools=["m1"], auto_apply=True)
    st = o.read_fleet_state()
    decs = o.maybe_restart_train_for_proposed_args(st)
    assert len(decs) == 1
    # No train_launch_cmd.json → cannot respawn even with auto_apply.
    assert "train_launch_cmd.json missing" in decs[0].reason


def test_missing_pid_file_no_restart(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1", n_envs=4)
    _write_applied(
        tmp_path,
        "m1",
        {"machine_id": "m1", "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256}},
    )
    o = _orch(tmp_path, pools=["m1"], auto_apply=True)
    st = o.read_fleet_state()
    decs = o.maybe_restart_train_for_proposed_args(st)
    assert len(decs) == 1
    assert decs[0].applied is False
    assert "train_launch_cmd.json missing" in decs[0].reason


def test_missing_pid_restarts_when_launch_cmd_present(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1", n_envs=4)
    _write_applied(
        tmp_path,
        "m1",
        {"machine_id": "m1", "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256}},
    )
    d = tmp_path / "fleet" / "m1"
    (d / "train_launch_cmd.json").write_text(
        json.dumps(
            {
                "cmd": [sys.executable, "-c", "print(1)"],
                "env": {},
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    o = _orch(
        tmp_path,
        pools=["m1"],
        auto_apply=True,
        apply_cooldown_s=0.0,
        dry_run=False,
        reconfig_ack_timeout_s=0.01,
    )
    st = o.read_fleet_state()
    with (
        patch.object(fo, "_cleanup_fleet_train_processes_for_machine", return_value=[]),
        patch.object(
            FleetOrchestrator, "_respawn_train_from_launch_file", return_value=424242
        ),
    ):
        decs = o.maybe_restart_train_for_proposed_args(st)
    assert any(d.applied for d in decs)
    assert (d / "train.pid").read_text(encoding="utf-8").strip().split()[0] == "424242"


def test_missing_launch_cmd_no_restart(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1", n_envs=4)
    _write_applied(
        tmp_path,
        "m1",
        {"machine_id": "m1", "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256}},
    )
    d = tmp_path / "fleet" / "m1"
    (d / "train.pid").write_text("12345\n", encoding="utf-8")
    o = _orch(tmp_path, pools=["m1"], auto_apply=True)
    st = o.read_fleet_state()
    with patch.object(fo, "_train_pid_process_alive", return_value=True):
        decs = o.maybe_restart_train_for_proposed_args(st)
    assert len(decs) == 1
    assert decs[0].applied is False
    assert "train_launch_cmd.json missing" in decs[0].reason


def test_cooldown_blocks_apply(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1", n_envs=4)
    _write_applied(
        tmp_path,
        "m1",
        {"machine_id": "m1", "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256}},
    )
    d = tmp_path / "fleet" / "m1"
    (d / "train.pid").write_text("12345\n", encoding="utf-8")
    (d / "train_launch_cmd.json").write_text(
        json.dumps(
            {
                "cmd": [sys.executable, "-c", "print(1)"],
                "env": {},
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    o = _orch(tmp_path, pools=["m1"], auto_apply=True, apply_cooldown_s=600.0)
    st = o.read_fleet_state()
    t0 = 1_000_000.0
    with (
        patch.object(fo, "_train_pid_process_alive", return_value=True),
        patch.object(fo, "_terminate_train_process_tree"),
        patch.object(
            FleetOrchestrator, "_respawn_train_from_launch_file", return_value=99999
        ),
        patch.object(fo.time, "time", return_value=t0 + 100.0),
    ):
        decs = o.maybe_restart_train_for_proposed_args(st)
    assert len(decs) == 1
    assert decs[0].kind == "train_restart_suppressed"
    assert decs[0].details.get("suppress_reason") == "cooldown"
    assert decs[0].applied is False
    assert "cooldown" in decs[0].reason


def test_circuit_breaker_after_three_restarts(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1", n_envs=4)
    _write_applied(
        tmp_path,
        "m1",
        {"machine_id": "m1", "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256}},
    )
    d = tmp_path / "fleet" / "m1"
    (d / "train.pid").write_text("12345\n", encoding="utf-8")
    (d / "train_launch_cmd.json").write_text(
        json.dumps(
            {
                "cmd": [sys.executable, "-c", "print(1)"],
                "env": {},
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    o = _orch(tmp_path, pools=["m1"], auto_apply=True, apply_cooldown_s=1.0)
    st = o.read_fleet_state()
    now = time.time()
    o._train_restart_times_by_machine["m1"] = [now - 60.0, now - 120.0, now - 180.0]
    with patch.object(fo, "_train_pid_process_alive", return_value=True):
        decs = o.maybe_restart_train_for_proposed_args(st)
    assert any("circuit breaker tripped" in d.reason for d in decs)
    assert not any(d.applied for d in decs)


def test_bootstrap_grace_suppresses_hash_drift_restart(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1", n_envs=4)
    _write_applied(
        tmp_path,
        "m1",
        {"machine_id": "m1", "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256}},
    )
    d = tmp_path / "fleet" / "m1"
    (d / "train_launch_cmd.json").write_text(
        json.dumps(
            {
                "cmd": [sys.executable, "-c", "print(1)"],
                "env": {},
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    (d / "train.pid").write_text("12345\n", encoding="utf-8")
    o = _orch(
        tmp_path,
        pools=["m1"],
        auto_apply=True,
        dry_run=False,
        apply_cooldown_s=0.0,
        train_bootstrap_grace_s=3600.0,
        reconfig_ack_timeout_s=0.01,
    )
    st = o.read_fleet_state()
    t_anchor = 2_000_000.0
    applied_path = d / "applied_args.json"
    ad = json.loads(applied_path.read_text(encoding="utf-8"))
    ad["applied_at"] = t_anchor
    applied_path.write_text(json.dumps(ad) + "\n", encoding="utf-8")
    with patch.object(fo.time, "time", return_value=t_anchor + 60.0):
        with patch.object(fo, "_train_pid_process_alive", return_value=True):
            decs = o.maybe_restart_train_for_proposed_args(st)
    assert len(decs) == 1
    assert decs[0].kind == "train_restart_suppressed"
    assert decs[0].details.get("suppress_reason") == "bootstrap_grace"
    assert decs[0].applied is False


def test_zombie_heal_respawns_when_hashes_aligned_no_live_train(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1", n_envs=4)
    _write_applied(
        tmp_path,
        "m1",
        {"machine_id": "m1", "args": {"--n-envs": 4, "--n-steps": 512, "--batch-size": 256}},
    )
    d = tmp_path / "fleet" / "m1"
    (d / "train_launch_cmd.json").write_text(
        json.dumps(
            {
                "cmd": [sys.executable, "-c", "print(1)"],
                "env": {},
                "cwd": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    (d / "train.pid").write_text("999001\n", encoding="utf-8")
    o = _orch(tmp_path, pools=["m1"], auto_apply=True, dry_run=False)
    o.train_zombie_heal_cooldown_s = 0.0
    st = o.read_fleet_state()
    with (
        patch.object(fo, "list_fleet_train_pids_for_machine", return_value=[]),
        patch.object(fo, "_cleanup_fleet_train_processes_for_machine", return_value=[]),
        patch.object(
            FleetOrchestrator, "_respawn_train_from_launch_file", return_value=77777
        ),
    ):
        decs = o.maybe_heal_stale_train(st)
    assert len(decs) == 1
    assert decs[0].kind == "train_zombie_heal"
    assert decs[0].applied is True
    assert (d / "train.pid").read_text(encoding="utf-8").strip().split()[0] == "77777"


def test_applied_args_missing_skips_compare(tmp_path: Path) -> None:
    _write_proposed(tmp_path, "m1")
    o = _orch(tmp_path, pools=["m1"], auto_apply=True)
    st = o.read_fleet_state()
    decs = o.maybe_restart_train_for_proposed_args(st)
    assert len(decs) == 1
    assert "bootstrap must seed" in decs[0].reason
