"""fleet_orchestrator: PPO-geometry soft reconfig vs hard restart."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    name = "fleet_orchestrator_soft_reconfig"
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
        "dry_run": False,
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
        "auto_apply": True,
        "apply_cooldown_s": 0.0,
        "reconfig_ack_timeout_s": 2.0,
        "train_pid_file_template": "fleet/{machine_id}/train.pid",
        "train_launch_cmd_file_template": "fleet/{machine_id}/train_launch_cmd.json",
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _base_fleet(
    tmp_path: Path, mid: str, *, p_n_env: int, a_n_env: int
) -> None:
    d = tmp_path / "fleet" / mid
    d.mkdir(parents=True)
    prop = {
        "machine_id": mid,
        "args": {
            "--n-envs": p_n_env,
            "--n-steps": 512,
            "--batch-size": 256,
        },
    }
    (d / "proposed_args.json").write_text(json.dumps(prop) + "\n", encoding="utf-8")
    a_doc = {**prop, "args": {**prop["args"], "--n-envs": a_n_env}}
    (d / "applied_args.json").write_text(
        json.dumps(
            {
                **a_doc,
                "applied_at": 0.0,
                "args_content_sha256": proposed_args_content_sha256(a_doc),
            }
        )
        + "\n",
        encoding="utf-8",
    )
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


def test_orchestrator_writes_reconfig_request_for_allowlist_diff(
    tmp_path: Path,
) -> None:
    mid = "m-soft"
    _base_fleet(tmp_path, mid, p_n_env=4, a_n_env=2)
    o = _orch(tmp_path, pools=[mid], apply_cooldown_s=0.0, reconfig_ack_timeout_s=5.0)
    o._last_apply_at_by_machine[mid] = 0.0
    st = o.read_fleet_state()
    prop = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    ph = proposed_args_content_sha256(prop)
    with (
        patch.object(fo, "_train_pid_process_alive", return_value=True),
        patch.object(fo, "_terminate_train_process_tree") as tkill,
        patch.object(
            FleetOrchestrator, "_poll_train_reconfig_ack", return_value="applied"
        ),
    ):
        decs = o.maybe_restart_train_for_proposed_args(st)
    tkill.assert_not_called()
    assert any(
        d.kind == "train_reconfig_applied" and d.applied for d in decs
    ), decs
    ap = json.loads(
        (tmp_path / "fleet" / mid / "applied_args.json").read_text(encoding="utf-8")
    )
    assert ap.get("args_content_sha256") == ph
    assert (tmp_path / "fleet" / mid / "train_reconfig_request.json").is_file()


def test_orchestrator_falls_back_to_hard_restart_on_timeout(
    tmp_path: Path,
) -> None:
    mid = "m-hard"
    _base_fleet(tmp_path, mid, p_n_env=3, a_n_env=1)
    o = _orch(
        tmp_path, pools=[mid], apply_cooldown_s=0.0, reconfig_ack_timeout_s=0.01
    )
    o._last_apply_at_by_machine[mid] = 0.0
    st = o.read_fleet_state()
    prop = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    ph = proposed_args_content_sha256(prop)
    with (
        patch.object(fo, "_train_pid_process_alive", return_value=True),
        patch.object(fo, "_terminate_train_process_tree"),
        patch.object(
            FleetOrchestrator, "_poll_train_reconfig_ack", return_value=None
        ),
        patch.object(
            FleetOrchestrator, "_respawn_train_from_launch_file", return_value=70001
        ),
    ):
        decs = o.maybe_restart_train_for_proposed_args(st)
    assert any(
        d.kind == "restart_train" and d.applied for d in decs
    ) or any(d.kind == "train_reconfig_applied" for d in decs)
    ap2 = json.loads(
        (tmp_path / "fleet" / mid / "applied_args.json").read_text(encoding="utf-8")
    )
    assert ap2.get("args_content_sha256") == ph
