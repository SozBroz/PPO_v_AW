# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tools.mcts_escalator import (
    DEFAULT_THRESHOLDS,
    EscalatorAction,
    EscalatorCycleResult,
    EscalatorState,
    EscalatorThresholds,
    append_cycle_log,
    compute_sims_proposal,
    decide_action,
    read_state,
    write_state,
)


def _cycle(
    *,
    winrate: float = 0.55,
    baseline: float = 0.50,
    games: int = 250,
    ev: float = 0.65,
    desyncs: int = 0,
    sims: int = 16,
    ts: float = 1.0,
    wall: float = 0.01,
) -> EscalatorCycleResult:
    return EscalatorCycleResult(
        cycle_ts=ts,
        sims=sims,
        winrate_vs_pool=winrate,
        mcts_off_baseline=baseline,
        games_decided=games,
        explained_variance=ev,
        engine_desyncs_in_cycle=desyncs,
        wall_s_per_decision_p50=wall,
    )


def test_read_state_missing_file_defaults_to_16_sims():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "missing.json"
        st = read_state(p)
        assert st.current_sims == 16
        assert st.mcts_off_baseline == 0.0
        assert st.cycles_at_current_sims == 0


def test_cold_start_first_cycle_is_warming_hold():
    st = read_state(Path("/nonexistent/nope.json"))
    cy = _cycle()
    prop = decide_action(st, cy)
    assert prop.action == EscalatorAction.HOLD
    assert "warming up" in prop.reason
    assert prop.proposed_sims == 16
    assert prop.state_after.cycles_at_current_sims == 1


def test_second_cycle_positive_lift_doubles_to_32():
    st = EscalatorState(
        current_sims=16,
        mcts_off_baseline=0.50,
        last_double_at_ts=0.0,
        sims_plateau_at=None,
        cycles_at_current_sims=1,
    )
    cy = _cycle(winrate=0.55, baseline=0.50, games=250, ev=0.65)
    prop = decide_action(st, cy)
    assert prop.action == EscalatorAction.DOUBLE
    assert prop.proposed_sims == 32
    assert prop.state_after.current_sims == 32
    assert prop.state_after.cycles_at_current_sims == 0


def test_engine_desync_drops_to_off_regardless_of_metrics():
    st = EscalatorState(
        current_sims=32,
        mcts_off_baseline=0.4,
        last_double_at_ts=1.0,
        sims_plateau_at=None,
        cycles_at_current_sims=5,
    )
    cy = _cycle(winrate=0.9, baseline=0.1, games=999, ev=0.99, desyncs=1)
    prop = decide_action(st, cy)
    assert prop.action == EscalatorAction.DROP_TO_OFF
    assert prop.proposed_sims == 0
    assert prop.state_after.current_sims == 0


def test_ev_below_regress_hold_plateau():
    st = EscalatorState(
        current_sims=32,
        mcts_off_baseline=0.50,
        last_double_at_ts=0.0,
        sims_plateau_at=None,
        cycles_at_current_sims=1,
    )
    cy = _cycle(winrate=0.55, baseline=0.50, games=250, ev=0.50)
    prop = decide_action(st, cy)
    assert prop.action == EscalatorAction.HOLD
    assert prop.state_after.sims_plateau_at == 32
    assert "explained_variance" in prop.reason


def test_negative_lift_hold_plateau():
    st = EscalatorState(
        current_sims=32,
        mcts_off_baseline=0.50,
        last_double_at_ts=0.0,
        sims_plateau_at=None,
        cycles_at_current_sims=1,
    )
    cy = _cycle(winrate=0.48, baseline=0.50, games=250, ev=0.70)
    prop = decide_action(st, cy)
    assert prop.action == EscalatorAction.HOLD
    assert prop.state_after.sims_plateau_at == 32
    assert "below baseline" in prop.reason


def test_at_128_positive_lift_stop_ask_operator():
    st = EscalatorState(
        current_sims=128,
        mcts_off_baseline=0.50,
        last_double_at_ts=99.0,
        sims_plateau_at=None,
        cycles_at_current_sims=1,
    )
    cy = _cycle(winrate=0.55, baseline=0.50, games=250, ev=0.65, sims=128)
    prop = decide_action(st, cy)
    assert prop.action == EscalatorAction.STOP_ASK_OPERATOR
    assert prop.proposed_sims == 128


def test_at_128_never_auto_double_past_cap():
    st = EscalatorState(
        current_sims=128,
        mcts_off_baseline=0.50,
        last_double_at_ts=99.0,
        sims_plateau_at=None,
        cycles_at_current_sims=1,
    )
    cy = _cycle(winrate=0.80, baseline=0.50, games=900, ev=0.95, sims=128)
    prop = decide_action(st, cy)
    assert prop.action == EscalatorAction.STOP_ASK_OPERATOR
    assert prop.proposed_sims == 128


def test_json_roundtrip_bytes_stable():
    st = EscalatorState(
        current_sims=64,
        mcts_off_baseline=0.42,
        last_double_at_ts=123.456,
        sims_plateau_at=32,
        cycles_at_current_sims=2,
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "st.json"
        write_state(p, st)
        b1 = p.read_bytes()
        st2 = read_state(p)
        write_state(p, st2)
        b2 = p.read_bytes()
        assert b1 == b2


def test_append_cycle_log_three_lines_parseable():
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "mcts_escalator.jsonl"
        for i in range(3):
            append_cycle_log(
                log,
                EscalatorCycleResult(
                    cycle_ts=float(i),
                    sims=16,
                    winrate_vs_pool=0.5,
                    mcts_off_baseline=0.48,
                    games_decided=200 + i,
                    explained_variance=0.6,
                    engine_desyncs_in_cycle=0,
                    wall_s_per_decision_p50=0.02,
                ),
            )
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)


def test_decide_action_idempotent():
    st = EscalatorState(
        current_sims=16,
        mcts_off_baseline=0.5,
        last_double_at_ts=0.0,
        sims_plateau_at=None,
        cycles_at_current_sims=1,
    )
    cy = _cycle(winrate=0.53, baseline=0.50, games=200, ev=0.6)
    a = decide_action(st, cy, thresholds=DEFAULT_THRESHOLDS)
    b = decide_action(st, cy, thresholds=DEFAULT_THRESHOLDS)
    assert a.action == b.action
    assert a.proposed_sims == b.proposed_sims
    assert a.reason == b.reason
    assert a.state_after == b.state_after


def test_compute_sims_proposal_apply_writes_state_and_log(tmp_path: Path):
    st_path = tmp_path / "mcts_escalator_state.json"
    log_path = tmp_path / "mcts_escalator.jsonl"
    cy = _cycle(ts=10.0)
    p0 = compute_sims_proposal(st_path, cy, apply=True, log_path=log_path)
    assert p0.action == EscalatorAction.HOLD
    st1 = read_state(st_path)
    assert st1.cycles_at_current_sims == 1
    assert log_path.is_file()
    assert len(log_path.read_text().strip().splitlines()) == 1


def test_min_cycles_before_double_respected():
    th = EscalatorThresholds(min_cycles_at_sims_before_double=3)
    st = EscalatorState(16, 0.5, 0.0, None, cycles_at_current_sims=2)
    cy = _cycle(winrate=0.60, baseline=0.50, games=300, ev=0.7)
    prop = decide_action(st, cy, thresholds=th)
    assert prop.action == EscalatorAction.HOLD
    assert "warming up" in prop.reason
