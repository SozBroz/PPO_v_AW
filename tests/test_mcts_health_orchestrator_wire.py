"""Phase 11d: fleet orchestrator reads mcts_health.json (audit only)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    spec = importlib.util.spec_from_file_location("fleet_orchestrator_mh", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["fleet_orchestrator_mh"] = m
    spec.loader.exec_module(m)
    return m


fo = _fo()
FleetOrchestrator = fo.FleetOrchestrator
read_mcts_health = fo.read_mcts_health


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
        "mcts_health_refresh_every_ticks": 0,
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _good_verdict() -> dict:
    return {
        "machine_id": "pc-b",
        "measured_at": "2099-01-01T00:00:00Z",
        "games_in_window": 200,
        "metrics": {
            "games_in_window": 200,
            "capture_sense_score": 0.7,
            "avg_capture_completions_per_game": 2.0,
            "avg_terrain_usage_score": 0.6,
            "army_value_lead_pos_rate": 0.6,
            "win_rate": 0.5,
            "avg_episode_length_turns": 30.0,
            "early_resign_rate": 0.0,
        },
        "pass_capture": True,
        "pass_terrain": True,
        "pass_army_value": True,
        "pass_episode_quality": True,
        "pass_overall": True,
        "proposed_mcts_mode": "eval_only",
        "proposed_mcts_sims": 16,
        "reasoning": "fixture",
    }


def test_read_mcts_health_none_when_missing(tmp_path: Path) -> None:
    assert read_mcts_health("nope", tmp_path) is None


def test_read_mcts_health_parses_fresh(tmp_path: Path) -> None:
    p = tmp_path / "fleet" / "pc-b" / "mcts_health.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {**_good_verdict(), "measured_at": "2099-01-01T00:00:00Z"}
        ),
        encoding="utf-8",
    )
    v = read_mcts_health("pc-b", tmp_path)
    assert v is not None
    assert v.proposed_mcts_mode == "eval_only"
    assert v.proposed_mcts_sims == 16


def test_read_mcts_health_stale_is_off(tmp_path: Path) -> None:
    p = tmp_path / "fleet" / "m1" / "mcts_health.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _good_verdict()
    payload["measured_at"] = "2000-01-01T00:00:00Z"
    p.write_text(json.dumps(payload), encoding="utf-8")
    v = read_mcts_health("m1", tmp_path)
    assert v is not None
    assert v.proposed_mcts_mode == "off"
    assert v.proposed_mcts_sims == 0
    assert v.pass_overall is False
    assert v.reasoning == "stale verdict"


def test_mcts_health_row_appears_in_tick_audit(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints").mkdir(parents=True)
    (tmp_path / "checkpoints" / "pool" / "pc-b").mkdir(parents=True)
    fp = tmp_path / "fleet" / "pc-b" / "mcts_health.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    payload = _good_verdict()
    payload["measured_at"] = "2099-02-01T00:00:00Z"
    fp.write_text(json.dumps(payload), encoding="utf-8")

    o = _orch(tmp_path, pools=["pc-b"], curator_min_age_minutes=0.0)
    o.tick()
    lines = o.audit_log.read_text(encoding="utf-8").strip().splitlines()
    jrows = [json.loads(L) for L in lines]
    mh = [r for r in jrows if r.get("kind") == "mcts_health"]
    assert len(mh) == 1
    assert mh[0]["machine_id"] == "pc-b"
    assert mh[0]["applied"] is False
    assert "verdict" in mh[0]["details"]


def test_mcts_health_stale_in_audit(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints").mkdir(parents=True)
    (tmp_path / "checkpoints" / "pool" / "z").mkdir(parents=True)
    q = tmp_path / "fleet" / "z" / "mcts_health.json"
    q.parent.mkdir(parents=True, exist_ok=True)
    w = {**_good_verdict(), "measured_at": "1990-01-15T00:00:00Z", "machine_id": "z"}
    q.write_text(json.dumps(w), encoding="utf-8")

    o = _orch(tmp_path, pools=["z"], curator_min_age_minutes=0.0)
    o.tick()
    jrows = [
        json.loads(L)
        for L in o.audit_log.read_text(encoding="utf-8").strip().splitlines()
    ]
    mh = [r for r in jrows if r.get("kind") == "mcts_health"][0]
    assert mh["details"]["verdict"]["proposed_mcts_mode"] == "off"
    assert mh["details"]["verdict"]["reasoning"] == "stale verdict"
