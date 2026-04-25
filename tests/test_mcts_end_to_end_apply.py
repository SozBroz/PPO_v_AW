# -*- coding: utf-8 -*-
"""Phase 11 Slice F — end-to-end ``orchestrator.tick()`` apply integration.

Two scenarios:

1. **Health-gate apply path**: A passing ``mcts_health.json`` plus
   ``--mcts-mode``-free ``proposed_args.json`` and a stale
   ``applied_args.json`` should — within a single ``tick()`` — merge
   the MCTS args, restart ``train.py`` (mocked PID/launch), and
   refresh ``applied_args.json`` so its ``args_content_sha256`` matches
   the new proposed.
2. **Escalator apply path**: With a baseline file present and a synthetic
   eval verdict, two ticks should drive the sim budget from 16 → 32
   via ``mcts_escalator_double``.

These tests are intentionally heavy on real file I/O and short on
mocks; the only mocks are the three ``subprocess`` / ``psutil`` shims
already used by ``tests/test_orchestrator_auto_apply.py``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    name = "fleet_orchestrator_e2e_apply"
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
        "repo_root": tmp_path,
        "keep_newest": 8,
        "keep_top_winrate": 12,
        "keep_diversity": 4,
        "curator_min_age_minutes": 0.0,
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
        "train_pid_file_template": "fleet/{machine_id}/train.pid",
        "train_launch_cmd_file_template": "fleet/{machine_id}/train_launch_cmd.json",
        "curriculum_enabled": False,
        "mcts_health_refresh_every_ticks": 0,
        "host_machine_id": "no-such-host",
        "enable_mcts_here": True,
        "mcts_gate_required_consecutive": 1,
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _seed_train_pidfile_and_launch(fleet_dir: Path) -> None:
    (fleet_dir / "train.pid").write_text("12345\n", encoding="utf-8")
    (fleet_dir / "train_launch_cmd.json").write_text(
        json.dumps(
            {
                "cmd": [sys.executable, "-c", "print(1)"],
                "env": {},
                "cwd": str(fleet_dir),
            }
        ),
        encoding="utf-8",
    )


def _write_probe(tmp_path: Path, mid: str) -> None:
    fleet = tmp_path / "fleet" / mid
    fleet.mkdir(parents=True, exist_ok=True)
    (fleet / "probe.json").write_text(
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


def _write_health_pass(tmp_path: Path, mid: str, *, mode: str, sims: int) -> None:
    """Write a passing mcts_health.json with fresh measured_at."""
    fleet = tmp_path / "fleet" / mid
    fleet.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {
        "machine_id": mid,
        "measured_at": now,
        "games_in_window": 80,
        "metrics": {
            "games_in_window": 80,
            "capture_sense_score": 0.6,
            "avg_capture_completions_per_game": 2.5,
            "avg_terrain_usage_score": 0.6,
            "army_value_lead_pos_rate": 0.6,
            "win_rate": 0.55,
            "avg_episode_length_turns": 32.0,
            "early_resign_rate": 0.05,
        },
        "pass_capture": True,
        "pass_terrain": True,
        "pass_army_value": True,
        "pass_episode_quality": True,
        "pass_overall": True,
        "proposed_mcts_mode": mode,
        "proposed_mcts_sims": int(sims),
        "reasoning": "synthetic test verdict",
    }
    (fleet / "mcts_health.json").write_text(json.dumps(body), encoding="utf-8")


def _seed_baseline_applied_args(tmp_path: Path, mid: str) -> dict:
    """Bootstrap applied_args.json with the args ``propose_from_probe`` will emit."""
    import tools.propose_train_args as pt

    probe = json.loads(
        (tmp_path / "fleet" / mid / "probe.json").read_text(encoding="utf-8")
    )
    base_doc = pt.propose_from_probe(probe)
    args = dict(base_doc.get("args") or {})
    h = proposed_args_content_sha256({"args": args})
    applied = {
        "machine_id": mid,
        "args": args,
        "applied_at": 1.0,
        "args_content_sha256": h,
    }
    (tmp_path / "fleet" / mid / "applied_args.json").write_text(
        json.dumps(applied), encoding="utf-8"
    )
    return applied


# ---------------------------------------------------------------------------
# Health-gate end-to-end apply
# ---------------------------------------------------------------------------


def test_health_gate_end_to_end_merges_and_restarts(tmp_path: Path) -> None:
    mid = "m-e2e"
    _write_probe(tmp_path, mid)
    _seed_baseline_applied_args(tmp_path, mid)
    _write_health_pass(tmp_path, mid, mode="eval_only", sims=16)
    _seed_train_pidfile_and_launch(tmp_path / "fleet" / mid)
    (tmp_path / "checkpoints").mkdir(parents=True)

    o = _orch(
        tmp_path,
        pools=[mid],
        auto_apply=True,
        apply_cooldown_s=0.0,
        host_machine_id=mid,
        enable_mcts_here=True,
        mcts_gate_required_consecutive=1,
        mcts_health_refresh_every_ticks=0,  # use our seeded mcts_health.json
    )

    with (
        patch.object(fo, "_train_pid_process_alive", return_value=True),
        patch.object(fo, "_terminate_train_process_tree"),
        patch.object(
            FleetOrchestrator,
            "_respawn_train_from_launch_file",
            return_value=99999,
        ),
    ):
        decisions = o.tick()

    proposed = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert proposed["args"]["--mcts-mode"] == "eval_only"
    assert int(proposed["args"]["--mcts-sims"]) == 16

    applied = json.loads(
        (tmp_path / "fleet" / mid / "applied_args.json").read_text(encoding="utf-8")
    )
    new_h = proposed_args_content_sha256(proposed)
    assert new_h is not None
    assert applied["args_content_sha256"] == new_h
    assert int(applied["args"]["--mcts-sims"]) == 16

    restart_rows = [d for d in decisions if d.kind == "restart_train"]
    assert len(restart_rows) == 1
    assert restart_rows[0].applied is True
    assert restart_rows[0].details.get("new_pid") == 99999

    # Pid file overwritten with the (mocked) respawn pid
    assert (tmp_path / "fleet" / mid / "train.pid").read_text(
        encoding="utf-8"
    ).strip() == "99999"


# ---------------------------------------------------------------------------
# Escalator end-to-end apply
# ---------------------------------------------------------------------------


def test_escalator_end_to_end_doubles_sims_on_second_tick(tmp_path: Path) -> None:
    """Two ticks: tick 1 = HOLD warm-up, tick 2 = DOUBLE 16→32 written to proposed_args."""
    mid = "m-esc"
    _write_probe(tmp_path, mid)
    _seed_baseline_applied_args(tmp_path, mid)
    _write_health_pass(tmp_path, mid, mode="eval_only", sims=16)
    _seed_train_pidfile_and_launch(tmp_path / "fleet" / mid)
    (tmp_path / "checkpoints").mkdir(parents=True)

    from tools.mcts_baseline import MctsOffBaseline, write_baseline
    from tools.mcts_escalator import EscalatorCycleResult

    write_baseline(
        MctsOffBaseline(
            schema_version=1,
            machine_id=mid,
            captured_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            checkpoint_zip=f"checkpoints/pool/{mid}/latest.zip",
            checkpoint_zip_sha256="e" * 64,
            games_decided=200,
            winrate_vs_pool=0.40,
            mcts_mode="off",
            source="tools/capture_mcts_baseline.py",
        ),
        tmp_path / "fleet" / mid,
    )

    # Strong winrate lift + EV well above default 0.6 threshold so the
    # escalator's ROI gates pass on tick 2.
    def _strong_cycle(machine_id, shared_root, baseline, **_kw):
        sims = 16
        try:
            doc = json.loads(
                (Path(shared_root) / "fleet" / str(machine_id) / "proposed_args.json")
                .read_text(encoding="utf-8")
            )
            sims = int(doc.get("args", {}).get("--mcts-sims", 16))
        except Exception:
            pass
        return EscalatorCycleResult(
            cycle_ts=1.0,
            sims=sims,
            winrate_vs_pool=0.62,
            mcts_off_baseline=float(baseline.winrate_vs_pool),
            games_decided=400,
            explained_variance=0.8,
            engine_desyncs_in_cycle=0,
            wall_s_per_decision_p50=0.01,
        )

    o = _orch(
        tmp_path,
        pools=[mid],
        auto_apply=True,
        apply_cooldown_s=0.0,
        host_machine_id=mid,
        enable_mcts_here=True,
        mcts_gate_required_consecutive=1,
        mcts_health_refresh_every_ticks=0,
    )

    with (
        patch.object(
            fo._mcts_eval_summary, "build_cycle_result", side_effect=_strong_cycle
        ),
        patch.object(fo, "_train_pid_process_alive", return_value=True),
        patch.object(fo, "_terminate_train_process_tree"),
        patch.object(
            FleetOrchestrator,
            "_respawn_train_from_launch_file",
            return_value=99999,
        ),
    ):
        d1 = o.tick()
        # Tick 1: warm-up — escalator HOLDs, sims stay at 16 in proposed_args.
        proposed_after_t1 = json.loads(
            (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
        )
        assert int(proposed_after_t1["args"]["--mcts-sims"]) == 16
        assert any(d.kind == "mcts_escalator_hold" for d in d1)

        d2 = o.tick()

    proposed_after_t2 = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert int(proposed_after_t2["args"]["--mcts-sims"]) == 32
    assert proposed_after_t2["args"]["--mcts-mode"] == "eval_only"

    doubles = [d for d in d2 if d.kind == "mcts_escalator_double"]
    assert len(doubles) == 1
    assert doubles[0].applied is True
    assert doubles[0].details["proposed_sims"] == 32
