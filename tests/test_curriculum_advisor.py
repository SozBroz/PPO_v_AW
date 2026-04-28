"""tools/curriculum_advisor: stages, metrics, state, probe-key stripping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.curriculum_advisor import (
    DEFAULT_SCHEDULE,
    CompetenceMetrics,
    CurriculumState,
    FLAG_PRESENT,
    PROBE_OWNED_KEYS,
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
    done: bool = True,
) -> dict:
    d: dict[str, Any] = {
        "machine_id": mid,
        "turns": turns,
        "done": done,
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
    assert normalize_curriculum_stage_name("stage_a") == "stage_a0_capture_decay"
    assert normalize_curriculum_stage_name("stage_b") == "stage_b0_capture_decay"
    assert normalize_curriculum_stage_name("stage_d") == "stage_d0_gl_std_map_pool_stub"
    assert (
        normalize_curriculum_stage_name("stage_d_gl_std_map_pool_t3")
        == "stage_d0_gl_std_map_pool_stub"
    )
    assert normalize_curriculum_stage_name("stage_d_self_play_pure") == (
        "stage_f0_full_random_stub"
    )
    assert normalize_curriculum_stage_name("stage_g") == "stage_g_mcts_eval_ready"


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
    assert st.current_stage_name == "stage_d0_gl_std_map_pool_stub"


def test_cold_start_stage_a(tmp_path: Path) -> None:
    """First schedule row is family A decay (scaffolded)."""
    log = tmp_path / "game_log.jsonl"
    _write_log(log, [])
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].stage_id,
        games_observed_in_stage=0,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop = compute_proposal(log, st, machine_id="m1")
    assert prop.stage_name == DEFAULT_SCHEDULE[0].stage_id
    assert prop.args_overrides["--cold-opponent"] == "greedy_capture"
    assert prop.args_overrides.get("--capture-move-gate") == FLAG_PRESENT


def test_metrics_insufficient_no_advance(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    rows = [_row(mid="m1", cap_p0=0.0, terrain=0.0, winner=1, turns=10)] * 250
    _write_log(log, rows)
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].stage_id,
        games_observed_in_stage=250,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop, st2 = compute_proposal_stable(log, st, machine_id="m1", window_games=100)
    assert prop.stage_name == DEFAULT_SCHEDULE[0].stage_id
    assert st2.current_stage_name == DEFAULT_SCHEDULE[0].stage_id


def test_advance_when_gates_met(tmp_path: Path) -> None:
    """PROMOTE from stage_a1_capture_clean uses decide_stage_transition (zeros on row)."""
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
        current_stage_name="stage_a1_capture_clean",
        games_observed_in_stage=200,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=220,
    )
    prop, st2 = compute_proposal_stable(log, st, machine_id="m1", window_games=100)
    assert prop.stage_name == "stage_a1_capture_clean"
    assert prop.args_overrides["--cold-opponent"] == "greedy_capture"
    assert st2.current_stage_name == "stage_b0_capture_decay"


def test_idempotent_proposal_same_state(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    _write_log(log, [_row(mid="m1")] * 80)
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].stage_id,
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


def test_unknown_stage_id_resolves_to_first_schedule_row(tmp_path: Path) -> None:
    log = tmp_path / "game_log.jsonl"
    _write_log(log, [_row(mid="m1")] * 80)
    st = CurriculumState(
        current_stage_name="bogus_unknown",
        games_observed_in_stage=999,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop = compute_proposal(log, st, machine_id="m1")
    assert prop.stage_name == DEFAULT_SCHEDULE[0].stage_id


def test_terminal_stage_f2_has_no_scheduled_successor(tmp_path: Path) -> None:
    """Last YAML row is stage_f2_full_random_clean; PROMOTE leaves state on f2 if _find_next_stage is None."""
    log = tmp_path / "game_log.jsonl"
    _write_log(log, [_row(mid="m1")] * 50)
    st = CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[-1].stage_id,
        games_observed_in_stage=10,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
    )
    prop = compute_proposal(log, st, machine_id="m1")
    assert DEFAULT_SCHEDULE[-1].stage_id == "stage_f2_full_random_clean"
    assert prop.stage_name == DEFAULT_SCHEDULE[-1].stage_id


def test_stage_a1_blocked_median_captures(tmp_path: Path) -> None:
    """Promotion from a1_clean requires min_captures_by_day5_p50 (4): cap_p0 3 stays on a1."""
    log = tmp_path / "game_log.jsonl"
    borderline = _row(mid="m1", cap_p0=3.0, first_p0_capture_p0_step=8.0)
    _write_log(log, [borderline] * 220)
    st = CurriculumState(
        current_stage_name="stage_a1_capture_clean",
        games_observed_in_stage=200,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=220,
    )
    prop, st2 = compute_proposal_stable(log, st, machine_id="m1", window_games=100)
    assert prop.stage_name == "stage_a1_capture_clean"
    assert st2.current_stage_name == "stage_a1_capture_clean"


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
    ma, _ = compute_metrics(log, window_games=10, machine_id="a")
    mb, _ = compute_metrics(log, window_games=10, machine_id="b")
    assert ma.capture_sense_score >= 0.9
    assert mb.capture_sense_score < 0.1


def test_probe_owned_keys_frozen() -> None:
    assert "--n-envs" in PROBE_OWNED_KEYS
