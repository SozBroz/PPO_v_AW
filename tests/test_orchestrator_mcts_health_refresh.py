"""Phase 11d Slice A: orchestrator periodically writes fleet/<id>/mcts_health.json."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    name = "fleet_orchestrator_mh_refresh"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


fo = _fo()
FleetOrchestrator = fo.FleetOrchestrator


def _orch(tmp_path: Path, *, pools: list[str], **kw: object) -> object:
    defaults: dict = {
        "shared_root": tmp_path,
        "pools": pools,
        "dry_run": True,
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
        "curriculum_enabled": False,
        "host_machine_id": "no-such-host",
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _good_row(mid: str) -> dict:
    return {
        "machine_id": mid,
        "turns": 35,
        "captures_completed_p0": 3.0,
        "terrain_usage_p0": 0.6,
        "winner": 0,
        "losses_hp": [1.0, 8.0],
    }


def _seed_logs(tmp_path: Path, mid: str, n: int = 60) -> Path:
    log_dir = tmp_path / "logs" / mid
    log_dir.mkdir(parents=True, exist_ok=True)
    gl = log_dir / "game_log.jsonl"
    gl.write_text(
        "\n".join(json.dumps(_good_row(mid)) for _ in range(n)) + "\n",
        encoding="utf-8",
    )
    return gl


def test_refresh_writes_mcts_health_json_per_machine(tmp_path: Path) -> None:
    mid = "ref-1"
    _seed_logs(tmp_path, mid, n=60)
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(tmp_path, pools=[mid])
    decs = o.refresh_mcts_health_documents(o.read_fleet_state())
    out = tmp_path / "fleet" / mid / "mcts_health.json"
    assert out.is_file()
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["machine_id"] == mid
    assert body["pass_overall"] is True
    assert any(
        d.kind == "mcts_health_refresh"
        and d.applied is True
        and d.machine_id == mid
        and d.details["window"] == 200
        and d.details["machine_id"] == mid
        for d in decs
    )


def test_refresh_skips_when_within_cadence_window(tmp_path: Path) -> None:
    mid = "ref-2"
    _seed_logs(tmp_path, mid, n=60)
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(
        tmp_path, pools=[mid], mcts_health_refresh_every_ticks=3
    )
    o._tick_counter = 1
    d1 = o.refresh_mcts_health_documents(o.read_fleet_state())
    assert any(x.kind == "mcts_health_refresh" and x.applied for x in d1)
    o._tick_counter = 2
    d2 = o.refresh_mcts_health_documents(o.read_fleet_state())
    assert d2 == []
    o._tick_counter = 4
    d3 = o.refresh_mcts_health_documents(o.read_fleet_state())
    assert any(x.kind == "mcts_health_refresh" and x.applied for x in d3)


def test_refresh_exception_does_not_crash_tick(tmp_path: Path) -> None:
    mid = "ref-3"
    _seed_logs(tmp_path, mid, n=60)
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(tmp_path, pools=[mid])
    with patch.object(
        fo._mcts_health, "compute_health", side_effect=RuntimeError("boom")
    ):
        decs = o.refresh_mcts_health_documents(o.read_fleet_state())
    assert any(
        d.kind == "mcts_health_refresh"
        and d.applied is False
        and d.machine_id == mid
        and "boom" in d.details["error"]
        for d in decs
    )


def test_refresh_runs_before_proposed_args_merge_in_tick(tmp_path: Path) -> None:
    """Tick chain: refresh writes a passing verdict, hysteresis at 1 → same-tick merge."""
    mid = "ref-4"
    _seed_logs(tmp_path, mid, n=60)
    (tmp_path / "fleet" / mid).mkdir(parents=True, exist_ok=True)
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
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(
        tmp_path,
        pools=[mid],
        dry_run=False,
        mcts_gate_required_consecutive=1,
        mcts_health_refresh_every_ticks=1,
    )
    o.tick()
    health = json.loads(
        (tmp_path / "fleet" / mid / "mcts_health.json").read_text(encoding="utf-8")
    )
    assert health["pass_overall"] is True
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--mcts-mode"] == health["proposed_mcts_mode"]
    assert int(doc["args"]["--mcts-sims"]) == int(health["proposed_mcts_sims"])
