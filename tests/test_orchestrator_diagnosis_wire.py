# -*- coding: utf-8 -*-
"""Phase 10h: fleet diagnosis wired into fleet_orchestrator tick (audit-only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    spec = importlib.util.spec_from_file_location("fleet_orchestrator_fd", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["fleet_orchestrator_fd"] = m
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
        "curriculum_enabled": False,
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _row(mid: str, winner: int, n_actions: float, ts: float) -> dict:
    return {
        "machine_id": mid,
        "turns": 25,
        "winner": winner,
        "n_actions": n_actions,
        "captures_completed_p0": 3.0,
        "opponent_type": "greedy_mix",
        "episode_wall_s": 10.0,
        "timestamp": ts,
    }


def test_tick_writes_diagnosis_and_audit_row(tmp_path: Path) -> None:
    mid = "keras-aux"
    (tmp_path / "checkpoints" / "pool" / mid).mkdir(parents=True)
    log = tmp_path / "logs" / mid / "game_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i in range(200):
        rows.append(_row(mid, 0 if i % 2 == 0 else 1, 100.0, float(i)))
    for i in range(200, 400):
        rows.append(_row(mid, 0 if i % 2 == 0 else 1, 165.0, float(i)))
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    o = _orch(tmp_path, pools=[mid], curator_min_age_minutes=0.0)
    o.tick()

    diag = tmp_path / "fleet" / mid / "diagnosis.json"
    assert diag.is_file()
    raw = json.loads(diag.read_text(encoding="utf-8"))
    assert raw.get("state") == "pathological"

    lines = o.audit_log.read_text(encoding="utf-8").strip().splitlines()
    jrows = [json.loads(L) for L in lines]
    fd = [r for r in jrows if r.get("kind") == "fleet_diagnosis"]
    assert len(fd) == 1
    assert fd[0]["details"].get("event") == "fleet_diagnosis"
    assert fd[0]["details"].get("state") == "pathological"


def test_proposed_args_untouched_when_no_probe(tmp_path: Path) -> None:
    mid = "aux-z"
    (tmp_path / "checkpoints" / "pool" / mid).mkdir(parents=True)
    prop = tmp_path / "fleet" / mid / "proposed_args.json"
    prop.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "args": {"--map-id": 1},
        "reasoning": "fixture",
        "auto_apply": False,
    }
    prop.write_text(json.dumps(body), encoding="utf-8")
    h_before = hashlib.sha256(prop.read_bytes()).hexdigest()

    log = tmp_path / "logs" / mid / "game_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "machine_id": mid,
                "turns": 10,
                "winner": 0,
                "n_actions": 50,
                "captures_completed_p0": 1,
                "opponent_type": "x",
                "episode_wall_s": 5.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    o = _orch(tmp_path, pools=[mid], curriculum_enabled=False)
    o.tick()
    h_after = hashlib.sha256(prop.read_bytes()).hexdigest()
    assert h_before == h_after
