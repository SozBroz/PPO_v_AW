"""tools/curriculum_advisor: stages, metrics, state, probe-key stripping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.curriculum_advisor import (
    DEFAULT_SCHEDULE,
    CompetenceMetrics,
    CurriculumStage,
    CurriculumState,
    FLAG_PRESENT,
    PROBE_OWNED_KEYS,
    classify_stage,
    compute_metrics,
    compute_proposal,
    compute_proposal_stable,
    normalize_curriculum_stage_name,
    read_state,
    write_state,
)


def _row(
    *,
    mid: str = "m1",
    cap_p0: float = 3.0,
    first_p0_capture_p0_step: float | None = 10.0,
    terrain: float = 0.6,
    winner: int = 0,
    turns: int = 40,
    p1_hp_loss: float = 10.0,
    p0_hp_loss: float = 2.0,
) -> dict:
    d: dict[str, Any] = {
        "machine_id": mid,
        "turns": turns,
        "captures_completed_p0": cap_p0,
        "terrain_usage_p0": terrain,
        "winner": winner,
        "losses_hp": [p0_hp_loss, p1_hp_loss],
    }
    if first_p0_capture_p0_step is not None:
        d["first_p0_capture_p0_step"] = first_p0_capture_p0_step
    return d


def _write_log(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_normalize_stage_d_shorthand() -> None:
    assert (
        normalize_curriculum_stage_name("stage_d")
        == "stage_d_gl_std_map_pool_t3"
    )
    assert normalize_curriculum_stage_name("stage_d_self_play_pure") == (
        "stage_f_self_play_pure"
    )
    assert (
        normalize_curriculum_stage_name("stage_g")
        == "stage_g_mcts_eval_ready"
    )


def test_read_state_accepts_stage_d_shorthand(tmp_path: Path) -> None:
    p = tmp_path / "curriculum_state.json"
    p.write_text(
        json.dumps(
            {
                "current_stage_name": "stage_d",
                "games_observed_in_stage": 12,
                "entered_stage_at_ts": 1.0,
                "last_proposal_ts": 2.0,
                "last_seen_finished_games": 100,
            }
        ),
        encoding="utf-8",
    )
    st = read_state(p)
    assert st.current_stage_name == "stage_d_gl_std_map_pool_t3"


def test_classify_respects_stage_d_shorthand() -> None:
    st = CurriculumState(
        current_stage_name="stage_d",
        games_observed_in_stage=0,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    m = CompetenceMetrics(
        games_in_window=0,
        capture_sense_score=0.0,
        terrain_usage_score=0.0,
        army_value_lead=0.0,
        win_rate=0.0,
        episode_quality=0.0,
    )
    name, reason = classify_stage(m, st, DEFAULT_SCHEDULE)
    assert name == "stage_d_gl_std_map_pool_t3"
    assert "stage_d_gl_std_map_pool_t3" in reason or "holding" in reason


def test_cold_start_stage_a(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    _write_log(log, [])
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].name,
        games_observed_in_stage=0,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop = compute_proposal(log, st, machine_id="m1")
    assert prop.stage_name == DEFAULT_SCHEDULE[0].name
    assert prop.args_overrides["--cold-opponent"] == "greedy_capture"
    assert prop.args_overrides.get("--capture-move-gate") == FLAG_PRESENT


def test_metrics_insufficient_no_advance(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    rows = [_row(mid="m1", cap_p0=0.0, terrain=0.0, winner=1, turns=10)] * 250
    _write_log(log, rows)
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].name,
        games_observed_in_stage=250,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop, st2 = compute_proposal_stable(log, st, machine_id="m1", window_games=100)
    assert prop.stage_name == DEFAULT_SCHEDULE[0].name
    assert st2.current_stage_name == DEFAULT_SCHEDULE[0].name


def test_advance_when_gates_met(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    good = _row(
        mid="m1",
        cap_p0=4.0,
        first_p0_capture_p0_step=10.0,
        terrain=0.6,
        winner=0,
        turns=35,
    )
    _write_log(log, [good] * 220)
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].name,
        games_observed_in_stage=200,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop, st2 = compute_proposal_stable(log, st, machine_id="m1", window_games=100)
    assert prop.stage_name == DEFAULT_SCHEDULE[1].name
    assert prop.args_overrides["--cold-opponent"] == "greedy_mix"
    assert st2.current_stage_name == DEFAULT_SCHEDULE[1].name


def test_idempotent_proposal_same_state(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    _write_log(log, [_row(mid="m1")] * 80)
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].name,
        games_observed_in_stage=50,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    p1 = compute_proposal(log, st, machine_id="m1")
    p2 = compute_proposal(log, st, machine_id="m1")
    assert p1.stage_name == p2.stage_name
    assert p1.args_overrides == p2.args_overrides
    assert p1.reason == p2.reason


def test_never_regresses_below_stage_a(tmp_path: Path) -> None:
    metrics = CompetenceMetrics(
        games_in_window=10,
        capture_sense_score=0.0,
        terrain_usage_score=0.0,
        army_value_lead=0.0,
        win_rate=0.0,
        episode_quality=0.0,
    )
    st = CurriculumState(
        current_stage_name="bogus_unknown",
        games_observed_in_stage=999,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    name, _reason = classify_stage(metrics, st, DEFAULT_SCHEDULE)
    assert name == DEFAULT_SCHEDULE[0].name


def test_terminal_stage_g_never_advances(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    _write_log(log, [_row(mid="m1")] * 50)
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[-1].name,
        games_observed_in_stage=10,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop = compute_proposal(log, st, machine_id="m1")
    assert prop.stage_name == DEFAULT_SCHEDULE[-1].name


def test_stage_a_blocked_median_captures(tmp_path: Path) -> None:
    """§10g: median captures must reach 4 before leaving stage A."""
    log = tmp_path / "game_log.jsonl"
    borderline = _row(mid="m1", cap_p0=3.0, first_p0_capture_p0_step=8.0)
    _write_log(log, [borderline] * 220)
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].name,
        games_observed_in_stage=200,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop, st2 = compute_proposal_stable(log, st, machine_id="m1", window_games=100)
    assert prop.stage_name == DEFAULT_SCHEDULE[0].name
    assert st2.current_stage_name == DEFAULT_SCHEDULE[0].name


def test_advance_f_to_g_includes_mcts_eval_only(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    good = _row(mid="m1", cap_p0=3.0, terrain=0.55, winner=0, turns=35)
    _write_log(log, [good] * 450)
    st = CurriculumState(
        current_stage_name="stage_f_self_play_pure",
        games_observed_in_stage=400,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop, st2 = compute_proposal_stable(
        log, st, machine_id="m1", window_games=100
    )
    assert prop.stage_name == "stage_g_mcts_eval_ready"
    assert prop.args_overrides.get("--mcts-mode") == "eval_only"
    assert prop.args_overrides.get("--mcts-sims") == 16
    assert st2.current_stage_name == "stage_g_mcts_eval_ready"


def test_strips_probe_owned_keys_from_custom_schedule(tmp_path: Path) -> None:
    bad = CurriculumStage(
        name="x",
        args_overrides={"--n-envs": 99, "--learner-greedy-mix": 0.1},
        advance_when=lambda m: False,
        min_games_in_stage=1,
        description="test",
    )
    log = tmp_path / "game_log.jsonl"
    _write_log(log, [_row(mid="m1")])
    st = read_state(tmp_path / "missing.json")
    prop, _ = compute_proposal_stable(log, st, schedule=[bad], machine_id="m1")
    assert "--n-envs" not in prop.args_overrides
    assert prop.args_overrides.get("--learner-greedy-mix") == 0.1


def test_state_json_roundtrip_bytes_identical(tmp_path: Path) -> None:
    p = tmp_path / "curriculum_state.json"
    st = read_state(p)
    write_state(p, st)
    b1 = p.read_bytes()
    write_state(p, read_state(p))
    b2 = p.read_bytes()
    assert b1 == b2


def test_compute_metrics_respects_machine_filter(tmp_path: Path) -> None:
    log = tmp_path / "gl.jsonl"
    _write_log(
        log,
        [
            _row(mid="a", cap_p0=3.0),
            _row(mid="b", cap_p0=0.0),
        ],
    )
    ma = compute_metrics(log, window_games=10, machine_id="a")
    mb = compute_metrics(log, window_games=10, machine_id="b")
    assert ma.capture_sense_score >= 0.9
    assert mb.capture_sense_score < 0.1


def test_probe_owned_keys_frozen() -> None:
    assert "--n-envs" in PROBE_OWNED_KEYS
