# -*- coding: utf-8 -*-
"""Phase 10h: fleet_diagnosis classifier unit tests."""
from __future__ import annotations

import json
from pathlib import Path

from tools.fleet_diagnosis import (
    DiagnosisState,
    classify,
    compute_diagnosis,
    compute_metrics,
    read_diagnosis,
    recommend_intervention,
    write_diagnosis,
)


def _row(
    *,
    mid: str = "m1",
    winner: int = 0,
    n_actions: float = 100.0,
    captures: float = 3.0,
    opponent: str = "greedy_capture",
    wall: float = 12.0,
    ts: float = 0.0,
) -> dict:
    return {
        "machine_id": mid,
        "turns": 28,
        "winner": winner,
        "n_actions": n_actions,
        "captures_completed_p0": captures,
        "opponent_type": opponent,
        "episode_wall_s": wall,
        "timestamp": ts,
    }


def _write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_empty_log_fresh(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    p.write_text("", encoding="utf-8")
    m = compute_metrics(p, "m1")
    st, reason = classify(m)
    assert st == DiagnosisState.FRESH
    assert "cold start" in reason.lower() or "0" in reason


def test_fifty_games_fresh(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows = [_row(winner=i % 2, ts=float(i)) for i in range(50)]
    _write_log(p, rows)
    m = compute_metrics(p, "m1")
    st, _ = classify(m)
    assert st == DiagnosisState.FRESH


def test_two_hundred_games_competent_no_curriculum(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows = [_row(winner=0, ts=float(i)) for i in range(200)]
    for i in range(0, 200, 7):
        rows[i] = _row(winner=1, ts=float(i))
    _write_log(p, rows)
    m = compute_metrics(p, "m1")
    st, reason = classify(m)
    assert st == DiagnosisState.COMPETENT
    assert m.winrate_recent_200 >= 0.65
    assert "winrate" in reason.lower() or "established" in reason.lower()


def test_regressing_winrate_drop(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows: list[dict] = []
    # oldest 100 in window: irrelevant filler
    for i in range(100):
        rows.append(_row(winner=0 if i % 2 == 0 else 1, ts=float(i)))
    # prior 200: 60% wins (winner 0)
    for i in range(100, 300):
        rows.append(_row(winner=0 if i % 5 != 0 else 1, ts=float(i)))
    # recent 200: 40% wins
    for i in range(300, 500):
        rows.append(_row(winner=0 if i % 5 == 0 else 1, ts=float(i)))
    _write_log(p, rows)
    m = compute_metrics(p, "m1")
    st, reason = classify(m)
    assert st == DiagnosisState.REGRESSING
    assert "winrate" in reason.lower()
    rec, _ = recommend_intervention(st, m, "m1", None, is_orchestrator_host=False)
    assert rec is not None
    assert "rollback" in rec.lower()


def test_stuck_flat_winrate_terminal_curriculum(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows = [_row(winner=0 if i % 2 == 0 else 1, ts=float(i)) for i in range(500)]
    _write_log(p, rows)
    curr = tmp_path / "curriculum_state.json"
    curr.write_text(
        json.dumps(
            {
                "current_stage_name": "stage_f_self_play_pure",
                "games_observed_in_stage": 600,
                "entered_stage_at_ts": 0.0,
                "last_proposal_ts": 0.0,
                "last_seen_finished_games": 500,
            }
        ),
        encoding="utf-8",
    )
    m = compute_metrics(
        p,
        "m1",
        curriculum_state_path=curr,
    )
    st, reason = classify(m)
    assert st == DiagnosisState.STUCK
    assert "flat" in reason.lower()
    rec, _ = recommend_intervention(st, m, "m1", None, is_orchestrator_host=False)
    assert rec is not None
    assert "ent-coef" in rec.lower() or "ent" in rec.lower()


def test_stuck_next_intervention_after_ent_bump(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows = [_row(winner=0 if i % 2 == 0 else 1, ts=float(i)) for i in range(500)]
    _write_log(p, rows)
    curr = tmp_path / "curriculum_state.json"
    curr.write_text(
        json.dumps(
            {
                "current_stage_name": "stage_f_self_play_pure",
                "games_observed_in_stage": 600,
                "entered_stage_at_ts": 0.0,
                "last_proposal_ts": 0.0,
                "last_seen_finished_games": 500,
            }
        ),
        encoding="utf-8",
    )
    hist = tmp_path / "intervention_history.json"
    hist.write_text(
        json.dumps({"entries": [{"kind": "stuck_ent_coef", "at_total_games": 200}]}),
        encoding="utf-8",
    )
    m = compute_metrics(
        p,
        "m1",
        curriculum_state_path=curr,
        intervention_history_path=hist,
    )
    st, _ = classify(m)
    assert st == DiagnosisState.STUCK
    rec, _ = recommend_intervention(
        st, m, "m1", hist, is_orchestrator_host=False
    )
    assert rec is not None
    assert "greedy" in rec.lower()


def test_pathological_n_actions_spike_flat_winrate(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows: list[dict] = []
    for i in range(200):
        rows.append(
            _row(
                winner=0 if i % 2 == 0 else 1,
                n_actions=100.0,
                ts=float(i),
            )
        )
    for i in range(200, 400):
        rows.append(
            _row(
                winner=0 if i % 2 == 0 else 1,
                n_actions=165.0,
                ts=float(i),
            )
        )
    _write_log(p, rows)
    m = compute_metrics(p, "m1")
    st, reason = classify(m)
    assert st == DiagnosisState.PATHOLOGICAL
    assert "n_actions" in reason.lower()
    rec, extra = recommend_intervention(
        st, m, "m1", None, is_orchestrator_host=False
    )
    assert rec is None
    assert "OPERATOR ALERT" in extra


def test_orchestrator_host_no_intervention(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows = [_row(winner=0 if i % 2 == 0 else 1, ts=float(i)) for i in range(500)]
    _write_log(p, rows)
    curr = tmp_path / "curriculum_state.json"
    curr.write_text(
        json.dumps(
            {
                "current_stage_name": "stage_f_self_play_pure",
                "games_observed_in_stage": 600,
                "entered_stage_at_ts": 0.0,
                "last_proposal_ts": 0.0,
                "last_seen_finished_games": 500,
            }
        ),
        encoding="utf-8",
    )
    m = compute_metrics(p, "m1", curriculum_state_path=curr)
    st, _ = classify(m)
    assert st == DiagnosisState.STUCK
    rec, extra = recommend_intervention(
        st, m, "pc-b", None, is_orchestrator_host=True
    )
    assert rec is None
    assert "pc-b" in extra.lower() or "orchestrator" in extra.lower()


def test_intervention_cooldown(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows = [_row(winner=0 if i % 2 == 0 else 1, ts=float(i)) for i in range(500)]
    _write_log(p, rows)
    curr = tmp_path / "curriculum_state.json"
    curr.write_text(
        json.dumps(
            {
                "current_stage_name": "stage_f_self_play_pure",
                "games_observed_in_stage": 600,
                "entered_stage_at_ts": 0.0,
                "last_proposal_ts": 0.0,
                "last_seen_finished_games": 500,
            }
        ),
        encoding="utf-8",
    )
    hist = tmp_path / "intervention_history.json"
    hist.write_text(
        json.dumps({"entries": [{"kind": "stuck_ent_coef", "at_total_games": 450}]}),
        encoding="utf-8",
    )
    m = compute_metrics(
        p,
        "m1",
        curriculum_state_path=curr,
        intervention_history_path=hist,
    )
    st, _ = classify(m)
    rec, extra = recommend_intervention(
        st, m, "m1", hist, is_orchestrator_host=False
    )
    assert rec is None
    assert "cooldown" in extra.lower()


def test_json_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows = [_row(winner=0, ts=float(i)) for i in range(80)]
    _write_log(p, rows)
    v = compute_diagnosis(p, "m1", is_orchestrator_host=False)
    out = tmp_path / "diagnosis.json"
    write_diagnosis(out, v)
    blob_a = out.read_bytes()
    v2 = read_diagnosis(out)
    assert v2 is not None
    write_diagnosis(out, v2)
    blob_b = out.read_bytes()
    assert blob_a == blob_b


def test_compute_diagnosis_merges_operator_alert_reason(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    rows: list[dict] = []
    for i in range(200):
        rows.append(_row(winner=0 if i % 2 == 0 else 1, n_actions=100.0, ts=float(i)))
    for i in range(200, 400):
        rows.append(_row(winner=0 if i % 2 == 0 else 1, n_actions=170.0, ts=float(i)))
    _write_log(p, rows)
    v = compute_diagnosis(p, "m1", is_orchestrator_host=False)
    assert v.state == DiagnosisState.PATHOLOGICAL
    assert "OPERATOR ALERT" in v.reason
    assert v.recommended_intervention is None
