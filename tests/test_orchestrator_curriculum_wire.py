"""fleet_orchestrator: curriculum merge into proposed_args + auto-apply interaction."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    name = "fleet_orchestrator_curriculum_wire"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


fo = _fo()
FleetOrchestrator = fo.FleetOrchestrator
proposed_args_content_sha256 = fo.proposed_args_content_sha256
proposed_document_body_sha256 = fo.proposed_document_body_sha256


def _orch(tmp_path: Path, *, pools: list[str], **kw: object) -> object:
    defaults: dict = {
        "shared_root": tmp_path,
        "pools": pools,
        "dry_run": False,
        "repo_root": tmp_path,
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
        "curriculum_enabled": True,
        "curriculum_window_games": 200,
        "curriculum_state_file_template": "fleet/{machine_id}/curriculum_state.json",
        "reconfig_ack_timeout_s": 120.0,
        "mcts_health_refresh_every_ticks": 0,
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _probe(mid: str) -> dict:
    return {
        "machine_id": mid,
        "probed_at": "2026-04-22T12:00:00Z",
        "cpu": {"physical_cores": 8},
        "ram": {"total_gb": 32.0},
    }


def _game_row(mid: str) -> dict:
    return {
        "machine_id": mid,
        "turns": 35,
        "done": True,
        "captures_completed_p0": 3.0,
        "first_p0_capture_p0_step": 10.0,
        "terrain_usage_p0": 0.55,
        "winner": 0,
        "losses_hp": [1.0, 8.0],
    }


def test_curriculum_stage_change_appends_fleet_curriculum_log(tmp_path: Path) -> None:
    """Stage transition appends one JSON line to logs/fleet_curriculum_changes.jsonl."""
    mid = "t-curr-chg"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    log_dir = tmp_path / "logs" / mid
    log_dir.mkdir(parents=True)
    gl = log_dir / "game_log.jsonl"
    row = {
        "machine_id": mid,
        "turns": 35,
        "done": True,
        "captures_completed_p0": 4.0,
        "first_p0_capture_p0_step": 10.0,
        "terrain_usage_p0": 0.6,
        "winner": 0,
        "losses_hp": [1.0, 8.0],
    }
    gl.write_text(
        "\n".join(json.dumps(row) for _ in range(220)) + "\n", encoding="utf-8"
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    (tmp_path / "fleet" / mid / "curriculum_state.json").write_text(
        json.dumps(
            {
                "current_stage_name": "stage_a_capture_bootstrap",
                "games_observed_in_stage": 200,
                "entered_stage_at_ts": 0.0,
                "last_proposal_ts": 0.0,
                "last_seen_finished_games": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    o = _orch(tmp_path, pools=[mid], curriculum_window_games=100)
    o.tick()
    log_path = tmp_path / "logs" / "fleet_curriculum_changes.jsonl"
    assert log_path.is_file()
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 1
    doc = json.loads(lines[0])
    assert doc["from_stage"] == "stage_a1_capture_clean"
    assert doc["to_stage"] == "stage_b0_capture_decay"
    assert doc["metrics"]["median_first_p0_capture_step"] == 10.0
    o.tick()
    lines2 = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines2) == 1


def test_tick_updates_proposed_args_with_stub_logs(tmp_path: Path) -> None:
    mid = "t-cur"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    log_dir = tmp_path / "logs" / mid
    log_dir.mkdir(parents=True)
    gl = log_dir / "game_log.jsonl"
    gl.write_text("\n".join(json.dumps(_game_row(mid)) for _ in range(60)) + "\n", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir(parents=True)

    o = _orch(tmp_path, pools=[mid])
    o.tick()
    prop_path = tmp_path / "fleet" / mid / "proposed_args.json"
    assert prop_path.is_file()
    doc = json.loads(prop_path.read_text(encoding="utf-8"))
    assert doc["args"]["--cold-opponent"] == "greedy_capture"
    assert "--capture-move-gate" in doc["args"]


def test_second_tick_same_logs_stable_sha256(tmp_path: Path) -> None:
    mid = "t-stable"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    log_dir = tmp_path / "logs" / mid
    log_dir.mkdir(parents=True)
    gl = log_dir / "game_log.jsonl"
    gl.write_text("\n".join(json.dumps(_game_row(mid)) for _ in range(40)) + "\n", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir(parents=True)

    o = _orch(tmp_path, pools=[mid])
    o.tick()
    p = tmp_path / "fleet" / mid / "proposed_args.json"
    h1 = hashlib.sha256(p.read_bytes()).hexdigest()
    o.tick()
    h2 = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h1 == h2


def test_refresh_proposed_pins_from_applied_when_proposed_missing_backend(
    tmp_path: Path,
) -> None:
    """If proposed lost --training-backend, recover from last applied_args."""
    mid = "t-pin-applied"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    log_dir = tmp_path / "logs" / mid
    log_dir.mkdir(parents=True)
    gl = log_dir / "game_log.jsonl"
    gl.write_text("\n".join(json.dumps(_game_row(mid)) for _ in range(60)) + "\n", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir(parents=True)
    bad_proposed = {
        "machine_id": mid,
        "args": {"--n-envs": 8},
    }
    (tmp_path / "fleet" / mid / "proposed_args.json").write_text(
        json.dumps(bad_proposed) + "\n", encoding="utf-8"
    )
    applied_body = {
        "machine_id": mid,
        "args": {
            "--n-envs": 14,
            "--training-backend": "async",
            "--live-games-id": [1, 2],
            "--live-snapshot-dir": str(tmp_path / "snap"),
        },
    }
    (tmp_path / "fleet" / mid / "applied_args.json").write_text(
        json.dumps(
            {
                **applied_body,
                "applied_at": 1.0,
                "args_content_sha256": proposed_args_content_sha256(applied_body),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    o = _orch(tmp_path, pools=[mid])
    o.tick()

    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--training-backend"] == "async"
    assert doc["args"]["--live-games-id"] == [1, 2]


def test_refresh_proposed_preserves_live_ppo_and_backend(tmp_path: Path) -> None:
    """Orchestrator ticks must not drop live PPO / async keys merged from start_solo_training."""
    mid = "t-preserve-live"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    log_dir = tmp_path / "logs" / mid
    log_dir.mkdir(parents=True)
    gl = log_dir / "game_log.jsonl"
    gl.write_text("\n".join(json.dumps(_game_row(mid)) for _ in range(60)) + "\n", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir(parents=True)
    snap = str((tmp_path / "replays" / "amarinner_my_games").resolve())
    pre = {
        "machine_id": mid,
        "args": {
            "--n-envs": 14,
            "--live-games-id": [1638496, 1638514],
            "--live-snapshot-dir": snap,
            "--training-backend": "async",
        },
    }
    (tmp_path / "fleet" / mid / "proposed_args.json").write_text(
        json.dumps(pre) + "\n", encoding="utf-8"
    )

    o = _orch(tmp_path, pools=[mid])
    o.tick()

    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--live-games-id"] == [1638496, 1638514]
    assert doc["args"]["--live-snapshot-dir"] == snap
    assert doc["args"]["--training-backend"] == "async"


def test_curriculum_disabled_no_curriculum_audit_rows(tmp_path: Path) -> None:
    mid = "t-off"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    (tmp_path / "logs" / mid).mkdir(parents=True)
    (tmp_path / "logs" / mid / "game_log.jsonl").write_text(
        json.dumps(_game_row(mid)) + "\n", encoding="utf-8"
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(tmp_path, pools=[mid], curriculum_enabled=False)
    d = o.tick()
    assert not any(x.kind == "curriculum_proposal" for x in d)


def test_auto_apply_proposed_hash_triggers_restart_mock(tmp_path: Path) -> None:
    mid = "t-auto"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    log_dir = tmp_path / "logs" / mid
    log_dir.mkdir(parents=True)
    gl = log_dir / "game_log.jsonl"
    gl.write_text("\n".join(json.dumps(_game_row(mid)) for _ in range(60)) + "\n", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir(parents=True)

    d_train = tmp_path / "fleet" / mid
    (d_train / "train.pid").write_text("424242\n", encoding="utf-8")
    (d_train / "train_launch_cmd.json").write_text(
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
        pools=[mid],
        auto_apply=True,
        apply_cooldown_s=1.0,
        reconfig_ack_timeout_s=0.01,
    )
    o.tick()
    prop = json.loads((d_train / "proposed_args.json").read_text(encoding="utf-8"))
    prop_h = proposed_args_content_sha256(prop)
    applied_doc = {
        **prop,
        "args": {**prop["args"], "--n-envs": 1},
        "applied_at": 1.0,
        "args_content_sha256": proposed_args_content_sha256(
            {**prop, "args": {**prop["args"], "--n-envs": 1}}
        ),
    }
    (d_train / "applied_args.json").write_text(
        json.dumps(applied_doc), encoding="utf-8"
    )

    with (
        patch.object(fo, "_train_pid_process_alive", return_value=True),
        patch.object(fo, "_terminate_train_process_tree"),
        patch.object(
            FleetOrchestrator, "_respawn_train_from_launch_file", return_value=77777
        ),
        patch.object(
            FleetOrchestrator, "_poll_train_reconfig_ack", return_value=None
        ),
        patch.object(fo.time, "time", return_value=10_000.0),
    ):
        st = o.read_fleet_state()
        decs = o.maybe_restart_train_for_proposed_args(st)

    assert any(d.applied for d in decs)
    applied2 = json.loads((d_train / "applied_args.json").read_text(encoding="utf-8"))
    assert applied2.get("args_content_sha256") == prop_h


def _orch_with_probe_and_log(tmp_path: Path) -> str:
    mid = "t-ovr"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(
            {
                "machine_id": mid,
                "probed_at": "2026-04-22T12:00:00Z",
                "cpu": {"physical_cores": 8},
                "ram": {"total_gb": 32.0},
            }
        ),
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs" / mid
    log_dir.mkdir(parents=True)
    gl = log_dir / "game_log.jsonl"
    gl.write_text(
        "\n".join(
            json.dumps(
                {
                    "machine_id": mid,
                    "turns": 35,
                    "captures_completed_p0": 3.0,
                    "first_p0_capture_p0_step": 10.0,
                    "terrain_usage_p0": 0.55,
                    "winner": 0,
                    "losses_hp": [1.0, 8.0],
                }
            )
            for _ in range(60)
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    return mid


def test_override_file_absent_no_change(tmp_path: Path) -> None:
    mid = _orch_with_probe_and_log(tmp_path)
    o = _orch(tmp_path, pools=[mid])
    o.tick()
    p = tmp_path / "fleet" / mid / "proposed_args.json"
    doc1 = json.loads(p.read_text(encoding="utf-8"))
    o.tick()
    doc2 = json.loads(p.read_text(encoding="utf-8"))
    assert doc1["args"] == doc2["args"]


def test_override_file_sparse_wins_for_listed_keys(tmp_path: Path) -> None:
    mid = _orch_with_probe_and_log(tmp_path)
    ovr = {
        "args": {"--n-envs": 12},
    }
    (tmp_path / "fleet" / mid / "operator_train_args_override.json").write_text(
        json.dumps(ovr) + "\n", encoding="utf-8"
    )
    o = _orch(tmp_path, pools=[mid])
    o.tick()
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--n-envs"] == 12
    d = o.tick()
    det = next(
        x
        for x in d
        if x.kind == "curriculum_proposal" and x.machine_id == mid
    )
    assert det.details.get("operator_overrides") == {"--n-envs": 12}
    assert "override:" in (doc.get("reasoning") or "")


def test_override_file_deleted_reverts_to_probe_default(tmp_path: Path) -> None:
    mid = _orch_with_probe_and_log(tmp_path)
    p_ovr = tmp_path / "fleet" / mid / "operator_train_args_override.json"
    p_ovr.write_text(json.dumps({"args": {"--n-envs": 12}}) + "\n", encoding="utf-8")
    o = _orch(tmp_path, pools=[mid])
    o.tick()
    assert (
        json.loads(
            (tmp_path / "fleet" / mid / "proposed_args.json").read_text(
                encoding="utf-8"
            )
        )["args"]["--n-envs"]
        == 12
    )
    p_ovr.unlink()
    o.tick()
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--n-envs"] < 12


def test_override_file_unknown_key_ignored_with_log(
    tmp_path: Path, caplog: Any
) -> None:
    mid = _orch_with_probe_and_log(tmp_path)
    (tmp_path / "fleet" / mid / "operator_train_args_override.json").write_text(
        json.dumps({"args": {"n_envs": 12, "--n-envs": 8}}) + "\n", encoding="utf-8"
    )
    o = _orch(tmp_path, pools=[mid])
    with caplog.at_level(logging.WARNING, logger=fo.__name__):
        o.tick()
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--n-envs"] == 8
    assert any("ignoring non-flag" in r.message for r in caplog.records)


def test_override_file_appears_in_audit_details(tmp_path: Path) -> None:
    mid = _orch_with_probe_and_log(tmp_path)
    (tmp_path / "fleet" / mid / "operator_train_args_override.json").write_text(
        json.dumps({"args": {"--n-envs": 9}}) + "\n", encoding="utf-8"
    )
    o = _orch(tmp_path, pools=[mid])
    d = o.tick()
    det = next(
        x
        for x in d
        if x.kind == "curriculum_proposal" and x.machine_id == mid
    )
    assert det.details.get("operator_overrides") == {"--n-envs": 9}
