# -*- coding: utf-8 -*-
"""Tests for curriculum sub-stage transition logic.

Covers:
- StagePhase enum
- ScaffoldConfig and StageConfig parsing from YAML
- decide_substage_transition() stub -> decay -> clean
- decide_stage_transition() promotion gate
- normalize_curriculum_stage_name()
- YAML round-trip (load + verify key stages)
- scaffold_at_episode() decay schedule indexing
- Anti-dependency rules (no promotion from stub/decay)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Ensure the repo root is on the path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.curriculum_advisor import (
    FLAG_PRESENT,
    StagePhase,
    StageDecision,
    SubStageDecision,
    ScaffoldConfig,
    SubStageCriteria,
    RollbackCriteria,
    PromotionCriteria,
    StageConfig,
    load_stages_yaml,
    normalize_curriculum_stage_name,
    _parse_criteria,
    _parse_scaffolds,
    _parse_rollback,
    _parse_promotion,
    _parse_phase,
    _stage_index_by_id,
    _find_next_stage,
    compute_metrics,
    decide_substage_transition,
    decide_stage_transition,
    CompetenceMetrics,
    ScaffoldMetrics,
    CurriculumState,
    compute_proposal_stable,
    read_state,
    write_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_yaml(tmp_path: Path) -> Path:
    yaml_content = """
stages:
  - stage_id: stage_a_capture_bootstrap
    stage_family: A
    stage_phase: clean
    distribution_change: none
    promotion_allowed: true
    map_pool_mode: fixed
    map_pool: [123858]
    co_pool_mode: fixed
    co_pool: [14]
    scaffolds:
      learner_greedy_mix: 0.30
      capture_move_gate: true
      opening_book_prob: 1.0
    cold_opponent: greedy_capture
    min_episodes: 200
    promotion_criteria:
      min_eval_games: 100
      min_terminal_rate: 0.60
      min_winrate: 0.45
    description: bootstrap

  - stage_id: stage_d0_gl_std_map_pool_stub
    stage_family: D
    stage_phase: stub
    distribution_change: map_pool
    promotion_allowed: false
    map_pool_mode: all_gl_std
    co_pool_mode: fixed
    co_pool: [14]
    scaffolds:
      learner_greedy_mix: 0.25
      capture_move_gate: true
      opening_book_prob: 0.50
    cold_opponent: greedy_mix
    min_episodes: 300
    stub_to_decay:
      min_episodes: 300
      min_terminal_rate: 0.20
      max_invalid_action_rate: 0.003
      max_first_capture_step_p50: 12
      min_captures_by_day5_p50: 4
      max_teacher_override_rate: 0.35
      max_capture_gate_intervention_rate: 0.45
      min_clean_probe_capture_sanity_rate: 0.60
    rollback_criteria:
      after_episodes: 500
      max_invalid_action_rate_above: 0.005
      first_capture_step_p50_above: 30
    description: map widening stub

  - stage_id: stage_d1_gl_std_map_pool_decay
    stage_family: D
    stage_phase: decay
    distribution_change: map_pool
    promotion_allowed: false
    map_pool_mode: all_gl_std
    co_pool_mode: fixed
    co_pool: [14]
    scaffolds:
      learner_greedy_mix: 0.25
      capture_move_gate: true
      opening_book_prob: 0.50
    cold_opponent: greedy_mix
    learner_greedy_mix_schedule: [0.25, 0.15, 0.05, 0.00]
    capture_move_gate_schedule: [true, true, false, false]
    opening_book_prob_schedule: [0.50, 0.25, 0.10, 0.00]
    min_episodes: 800
    decay_to_clean:
      min_episodes: 800
      learner_greedy_mix_at_end: 0.0
      capture_move_gate_at_end: false
      opening_book_prob_at_end: 0.0
      max_teacher_override_rate_final: 0.03
      max_capture_gate_intervention_rate_final: 0.05
      min_clean_probe_capture_sanity_rate: 0.75
    rollback_criteria:
      after_episodes: 1200
    description: map widening decay

  - stage_id: stage_d2_gl_std_map_pool_clean
    stage_family: D
    stage_phase: clean
    distribution_change: none
    promotion_allowed: true
    map_pool_mode: all_gl_std
    co_pool_mode: fixed
    co_pool: [14]
    scaffolds:
      learner_greedy_mix: 0.0
      capture_move_gate: false
      opening_book_prob: 0.0
    cold_opponent: greedy_mix
    min_episodes: 400
    promotion_criteria:
      min_eval_games: 100
      min_terminal_rate: 0.70
      max_max_env_steps_truncation_rate: 0.20
      min_winrate: 0.58
      max_first_capture_step_p50: 12
      min_captures_by_day5_p50: 4
    description: map widening clean
"""
    p = tmp_path / "stages.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    return p


@pytest.fixture
def schedule(minimal_yaml: Path) -> list[StageConfig]:
    return load_stages_yaml(minimal_yaml)


# ─────────────────────────────────────────────────────────────────────────────
# StagePhase enum
# ─────────────────────────────────────────────────────────────────────────────


class TestStagePhase:
    def test_phase_values(self):
        assert StagePhase.STUB.name == "STUB"
        assert StagePhase.DECAY.name == "DECAY"
        assert StagePhase.CLEAN.name == "CLEAN"
        assert StagePhase.PROMOTION_EVAL.name == "PROMOTION_EVAL"

    def test_phase_str(self):
        assert str(StagePhase.STUB) == "stub"
        assert str(StagePhase.DECAY) == "decay"
        assert str(StagePhase.CLEAN) == "clean"

    def test_parse_phase(self):
        from tools.curriculum_advisor import _parse_phase
        assert _parse_phase("stub") == StagePhase.STUB
        assert _parse_phase("decay") == StagePhase.DECAY
        assert _parse_phase("clean") == StagePhase.CLEAN
        assert _parse_phase("promotion_eval") == StagePhase.PROMOTION_EVAL
        assert _parse_phase("STUB") == StagePhase.STUB


# ─────────────────────────────────────────────────────────────────────────────
# StageConfig parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadYaml:
    def test_load_minimal_schedule(self, schedule):
        assert len(schedule) == 4
        ids = [s.stage_id for s in schedule]
        assert "stage_a_capture_bootstrap" in ids
        assert "stage_d0_gl_std_map_pool_stub" in ids
        assert "stage_d1_gl_std_map_pool_decay" in ids
        assert "stage_d2_gl_std_map_pool_clean" in ids

    def test_stage_family(self, schedule):
        for s in schedule:
            assert len(s.stage_family) == 1
        by_id = {s.stage_id: s for s in schedule}
        assert by_id["stage_a_capture_bootstrap"].stage_phase == StagePhase.CLEAN
        assert by_id["stage_d0_gl_std_map_pool_stub"].stage_phase == StagePhase.STUB
        assert by_id["stage_d1_gl_std_map_pool_decay"].stage_phase == StagePhase.DECAY
        assert by_id["stage_d2_gl_std_map_pool_clean"].stage_phase == StagePhase.CLEAN

    def test_stub_promotion_forbidden(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        assert stub.promotion_allowed is False

    def test_clean_promotion_allowed(self, schedule):
        clean = next(s for s in schedule if s.stage_phase == StagePhase.CLEAN)
        assert clean.promotion_allowed is True

    def test_scaffold_values(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        assert stub.scaffolds.learner_greedy_mix == 0.25
        assert stub.scaffolds.capture_move_gate is True
        assert stub.scaffolds.opening_book_prob == 0.50

    def test_decay_schedule_parsed(self, schedule):
        decay = next(s for s in schedule if s.stage_phase == StagePhase.DECAY)
        assert decay.learner_greedy_mix_schedule == [0.25, 0.15, 0.05, 0.00]
        assert decay.capture_move_gate_schedule == [True, True, False, False]
        assert decay.opening_book_prob_schedule == [0.50, 0.25, 0.10, 0.00]

    def test_substage_criteria_parsed(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        assert stub.stub_to_decay is not None
        assert stub.stub_to_decay.min_episodes == 300
        assert stub.stub_to_decay.min_captures_by_day5_p50 == 4
        assert stub.stub_to_decay.max_teacher_override_rate == 0.35

    def test_decay_to_clean_criteria(self, schedule):
        decay = next(s for s in schedule if s.stage_phase == StagePhase.DECAY)
        assert decay.decay_to_clean is not None
        assert decay.decay_to_clean.learner_greedy_mix_at_end == 0.0
        assert decay.decay_to_clean.capture_move_gate_at_end is False
        assert decay.decay_to_clean.min_clean_probe_capture_sanity_rate == 0.75

    def test_rollback_criteria(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        assert stub.rollback_criteria is not None
        assert stub.rollback_criteria.after_episodes == 500
        assert stub.rollback_criteria.max_invalid_action_rate_above == 0.005

    def test_promotion_criteria(self, schedule):
        clean = next(s for s in schedule if s.stage_id == "stage_d2_gl_std_map_pool_clean")
        assert clean.promotion_criteria is not None
        assert clean.promotion_criteria.min_terminal_rate == 0.70
        assert clean.promotion_criteria.min_winrate == 0.58

    def test_args_overrides_clean(self, schedule):
        clean = next(s for s in schedule if s.stage_id == "stage_d2_gl_std_map_pool_clean")
        overrides = clean.args_overrides()
        assert overrides["--learner-greedy-mix"] == 0.0
        assert overrides["--capture-move-gate"] is False
        assert overrides["--opening-book-prob"] == 0.0

    def test_args_overrides_stub(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        overrides = stub.args_overrides()
        assert overrides["--learner-greedy-mix"] == 0.25
        assert overrides["--capture-move-gate"] == FLAG_PRESENT
        assert overrides["--opening-book-prob"] == 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold decay schedule
# ─────────────────────────────────────────────────────────────────────────────


class TestScaffoldDecay:
    def test_scaffold_at_episode_stub_unchanged(self, schedule):
        """Stub stages always return their configured scaffolds."""
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        sc = stub.scaffold_at_episode(episode_idx=0)
        assert sc.learner_greedy_mix == 0.25
        assert sc.capture_move_gate is True
        assert sc.opening_book_prob == 0.50

    def test_scaffold_at_episode_decay_phases(self, schedule):
        """Decay stages return scaffolds at the correct phase boundary."""
        decay = next(s for s in schedule if s.stage_phase == StagePhase.DECAY)
        # 4 phases over 800 episodes = 200 episodes each
        # Phase 0: episodes 0-199
        sc0 = decay.scaffold_at_episode(episode_idx=0)
        assert sc0.learner_greedy_mix == 0.25
        assert sc0.capture_move_gate is True
        assert sc0.opening_book_prob == 0.50

        # Phase 1: episodes 200-399
        sc1 = decay.scaffold_at_episode(episode_idx=200)
        assert sc1.learner_greedy_mix == 0.15
        assert sc1.capture_move_gate is True

        # Phase 2: episodes 400-599
        sc2 = decay.scaffold_at_episode(episode_idx=400)
        assert sc2.learner_greedy_mix == 0.05
        assert sc2.capture_move_gate is False

        # Phase 3: episodes 600+
        sc3 = decay.scaffold_at_episode(episode_idx=600)
        assert sc3.learner_greedy_mix == 0.00
        assert sc3.capture_move_gate is False
        assert sc3.opening_book_prob == 0.00


# ─────────────────────────────────────────────────────────────────────────────
# Normalize curriculum stage name
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeStageName:
    def test_direct(self):
        assert normalize_curriculum_stage_name("stage_a_capture_bootstrap") == "stage_a_capture_bootstrap"
        assert normalize_curriculum_stage_name("stage_d0_gl_std_map_pool_stub") == "stage_d0_gl_std_map_pool_stub"

    def test_shorthand(self):
        assert normalize_curriculum_stage_name("stage_d") == "stage_d0_gl_std_map_pool_stub"
        assert normalize_curriculum_stage_name("stage_e") == "stage_e0_gl_mixed_co_stub"
        assert normalize_curriculum_stage_name("stage_f") == "stage_f0_full_random_stub"
        assert normalize_curriculum_stage_name("stage_a") == "stage_a_capture_bootstrap"

    def test_legacy_renames(self):
        assert normalize_curriculum_stage_name("stage_d_self_play_pure") == "stage_f_self_play_pure"
        assert normalize_curriculum_stage_name("stage_d_gl_std_map_pool_t3") == "stage_d0_gl_std_map_pool_stub"


# ─────────────────────────────────────────────────────────────────────────────
# Transition decision logic
# ─────────────────────────────────────────────────────────────────────────────


def _metrics(
    games_in_window: int = 100,
    win_rate: float = 0.50,
    median_first_p0_capture_step: float = 8.0,
    median_captures_completed_p0: float = 5.0,
    terminal_rate: float = 0.70,
) -> CompetenceMetrics:
    return CompetenceMetrics(
        games_in_window=games_in_window,
        capture_sense_score=0.6,
        terrain_usage_score=0.6,
        army_value_lead=0.5,
        win_rate=win_rate,
        episode_quality=0.8,
        median_first_p0_capture_step=median_first_p0_capture_step,
        median_captures_completed_p0=median_captures_completed_p0,
    )


def _s_metrics(
    invalid_action_rate: float = 0.001,
    teacher_override_rate: float = 0.10,
    capture_gate_intervention_rate: float = 0.15,
    clean_probe_capture_sanity_rate: float | None = 0.75,
    clean_probe_terminal_rate: float | None = 0.60,
    clean_probe_truncation_rate: float | None = 0.30,
    clean_probe_winrate: float | None = 0.50,
    terminal_rate: float = 0.70,
    max_env_steps_truncation_rate: float = 0.15,
) -> ScaffoldMetrics:
    return ScaffoldMetrics(
        teacher_override_rate=teacher_override_rate,
        capture_gate_intervention_rate=capture_gate_intervention_rate,
        opening_book_desync_rate=0.0,
        opening_book_used_count=0,
        opening_book_desync_count=0,
        invalid_action_rate=invalid_action_rate,
        terminal_rate=terminal_rate,
        max_env_steps_truncation_rate=max_env_steps_truncation_rate,
        clean_probe_capture_sanity_rate=clean_probe_capture_sanity_rate,
        clean_probe_terminal_rate=clean_probe_terminal_rate,
        clean_probe_truncation_rate=clean_probe_truncation_rate,
        clean_probe_winrate=clean_probe_winrate,
    )


class TestDecideSubstageTransitionStub:
    def test_stub_stay_too_few_games(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        m = _metrics(games_in_window=100)
        sm = _s_metrics()
        # stub_to_decay.min_episodes = 300; only 100 in window
        result = decide_substage_transition(m, sm, stub, games_in_stage=100)
        assert result == SubStageDecision.STAY

    def test_stub_advance_criteria_met(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        m = _metrics(games_in_window=300, median_first_p0_capture_step=8.0, median_captures_completed_p0=5.0)
        sm = _s_metrics(
            teacher_override_rate=0.20,
            capture_gate_intervention_rate=0.30,
            clean_probe_capture_sanity_rate=0.70,
        )
        result = decide_substage_transition(m, sm, stub, games_in_stage=300)
        assert result == SubStageDecision.ADVANCE_TO_DECAY

    def test_stub_stay_teacher_override_too_high(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        m = _metrics(games_in_window=300)
        sm = _s_metrics(teacher_override_rate=0.50)  # > 0.35 threshold
        result = decide_substage_transition(m, sm, stub, games_in_stage=300)
        assert result == SubStageDecision.STAY

    def test_stub_rollback_triggered(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        m = _metrics(games_in_window=600, median_first_p0_capture_step=35.0)  # > 30 threshold
        sm = _s_metrics()
        result = decide_substage_transition(m, sm, stub, games_in_stage=600)
        # after_episodes=500, first_capture_step_p50_above=30 → rollback
        assert result == SubStageDecision.ROLLBACK

    def test_stub_capture_gate_rate_too_high(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        m = _metrics(games_in_window=300)
        sm = _s_metrics(capture_gate_intervention_rate=0.60)  # > 0.45 threshold
        result = decide_substage_transition(m, sm, stub, games_in_stage=300)
        assert result == SubStageDecision.STAY


class TestDecideSubstageTransitionDecay:
    def test_decay_stay_too_few_games(self, schedule):
        decay = next(s for s in schedule if s.stage_phase == StagePhase.DECAY)
        m = _metrics(games_in_window=400)
        sm = _s_metrics()
        result = decide_substage_transition(m, sm, decay, games_in_stage=400)
        assert result == SubStageDecision.STAY

    def test_decay_advance_criteria_met(self, schedule):
        decay = next(s for s in schedule if s.stage_phase == StagePhase.DECAY)
        m = _metrics(games_in_window=800, median_first_p0_capture_step=8.0, median_captures_completed_p0=5.0)
        sm = _s_metrics(
            teacher_override_rate=0.01,
            capture_gate_intervention_rate=0.02,
            clean_probe_capture_sanity_rate=0.80,
            clean_probe_terminal_rate=0.60,
            clean_probe_truncation_rate=0.30,
        )
        result = decide_substage_transition(m, sm, decay, games_in_stage=800)
        assert result == SubStageDecision.ADVANCE_TO_CLEAN

    def test_decay_rollback_clean_probe_bad(self, schedule):
        decay = next(s for s in schedule if s.stage_phase == StagePhase.DECAY)
        # Override rollback criteria so clean_probe_capture_sanity_rate_below = 0.50
        # (instead of inheriting stub's non-matching criteria)
        # Also set decay_to_clean to have very low min_episodes so we pass criteria
        # but then hit rollback (which fires before criteria result is acted on).
        # Actually, rollback fires BEFORE criteria result in the function - but only
        # after criteria pass. So set criteria to pass, but make clean_probe bad.
        decay_with_rollback = StageConfig(
            stage_id=decay.stage_id,
            stage_family=decay.stage_family,
            stage_phase=decay.stage_phase,
            distribution_change=decay.distribution_change,
            promotion_allowed=decay.promotion_allowed,
            map_pool_mode=decay.map_pool_mode,
            map_pool=decay.map_pool,
            co_pool_mode=decay.co_pool_mode,
            co_pool=decay.co_pool,
            tier_mode=decay.tier_mode,
            tier=decay.tier,
            scaffolds=decay.scaffolds,
            cold_opponent=decay.cold_opponent,
            learner_greedy_mix_schedule=decay.learner_greedy_mix_schedule,
            capture_move_gate_schedule=decay.capture_move_gate_schedule,
            opening_book_prob_schedule=decay.opening_book_prob_schedule,
            min_episodes=100,  # very low so criteria pass
            stub_to_decay=decay.stub_to_decay,
            decay_to_clean=SubStageCriteria(
                min_episodes=100,  # pass this
                learner_greedy_mix_at_end=0.0,  # pass this
                capture_move_gate_at_end=False,  # pass this
                opening_book_prob_at_end=0.0,  # pass this
                max_teacher_override_rate_final=1.0,  # pass
                max_capture_gate_intervention_rate_final=1.0,  # pass
                min_clean_probe_capture_sanity_rate=0.80,  # PASS (so criteria return True)
            ),
            rollback_criteria=RollbackCriteria(
                after_episodes=1200,
                clean_probe_capture_sanity_rate_below=0.50,
            ),
            promotion_criteria=decay.promotion_criteria,
            description=decay.description,
        )
        # games_in_window=1200 >= min_episodes=100 -> criteria pass -> criteria returned True
        # -> ADVANCE_TO_CLEAN returned before rollback check! So rollback never fires.
        # FIX: use decay_to_clean with min_episodes > games_in_window so criteria FAIL
        # and STAY is returned, allowing the rollback block to be reached.
        decay_with_rollback = StageConfig(
            stage_id=decay.stage_id,
            stage_family=decay.stage_family,
            stage_phase=decay.stage_phase,
            distribution_change=decay.distribution_change,
            promotion_allowed=decay.promotion_allowed,
            map_pool_mode=decay.map_pool_mode,
            map_pool=decay.map_pool,
            co_pool_mode=decay.co_pool_mode,
            co_pool=decay.co_pool,
            tier_mode=decay.tier_mode,
            tier=decay.tier,
            scaffolds=decay.scaffolds,
            cold_opponent=decay.cold_opponent,
            learner_greedy_mix_schedule=decay.learner_greedy_mix_schedule,
            capture_move_gate_schedule=decay.capture_move_gate_schedule,
            opening_book_prob_schedule=decay.opening_book_prob_schedule,
            min_episodes=2000,  # exceeds games_in_window so criteria fail -> STAY
            stub_to_decay=decay.stub_to_decay,
            decay_to_clean=SubStageCriteria(
                min_episodes=2000,  # criteria fail -> STAY -> rollback checked
                learner_greedy_mix_at_end=0.0,
                capture_move_gate_at_end=False,
                opening_book_prob_at_end=0.0,
                max_teacher_override_rate_final=1.0,
                max_capture_gate_intervention_rate_final=1.0,
                min_clean_probe_capture_sanity_rate=0.80,
            ),
            rollback_criteria=RollbackCriteria(
                after_episodes=1200,
                clean_probe_capture_sanity_rate_below=0.50,
            ),
            promotion_criteria=decay.promotion_criteria,
            description=decay.description,
        )
        m = _metrics(games_in_window=1200)  # < 2000 -> criteria fail -> STAY
        sm = _s_metrics(clean_probe_capture_sanity_rate=0.40)  # < 0.50 -> rollback
        result = decide_substage_transition(m, sm, decay_with_rollback, games_in_stage=1200)
        assert result == SubStageDecision.ROLLBACK


class TestDecideStageTransition:
    # Use stage_d2 (scaffold-free clean) for all transition tests
    def _d2(self, schedule) -> StageConfig:
        return next(s for s in schedule if s.stage_id == "stage_d2_gl_std_map_pool_clean")

    def test_promote_clean_valid(self, schedule):
        clean = self._d2(schedule)
        m = _metrics(win_rate=0.62, median_first_p0_capture_step=8.0, median_captures_completed_p0=5.0)
        sm = _s_metrics(terminal_rate=0.75, invalid_action_rate=0.0005)
        result = decide_stage_transition(m, sm, clean)
        assert result == StageDecision.PROMOTE

    def test_no_promote_truncation_rate_high(self, schedule):
        clean = self._d2(schedule)
        m = _metrics(win_rate=0.65)
        # terminal_rate must pass min_terminal_rate=0.70 gate first
        sm = _s_metrics(terminal_rate=0.75, max_env_steps_truncation_rate=0.35)  # > 0.20 threshold
        result = decide_stage_transition(m, sm, clean)
        assert result == StageDecision.INVALID_EVAL_TRUNCATION

    def test_no_promote_winrate_low(self, schedule):
        clean = self._d2(schedule)
        m = _metrics(win_rate=0.45)  # < 0.58 threshold
        sm = _s_metrics(terminal_rate=0.75)
        result = decide_stage_transition(m, sm, clean)
        assert result == StageDecision.FAIL_WINRATE

    def test_no_promote_terminal_rate_low(self, schedule):
        clean = self._d2(schedule)
        m = _metrics(win_rate=0.62)
        sm = _s_metrics(terminal_rate=0.50)  # < 0.70 threshold
        result = decide_stage_transition(m, sm, clean)
        assert result == StageDecision.INVALID_EVAL_TERMINAL_RATE_LOW

    def test_no_promote_capture_delay(self, schedule):
        clean = self._d2(schedule)
        m = _metrics(win_rate=0.62, median_first_p0_capture_step=20.0)  # > 12 threshold
        sm = _s_metrics(terminal_rate=0.75)
        result = decide_stage_transition(m, sm, clean)
        assert result == StageDecision.FAIL_OPENING_CAPTURE_DELAY

    def test_clean_not_promotion_allowed(self, schedule):
        # The non-widening A stage is clean but should not allow promotion via stage_transition
        # (it's treated as a special bootstrap case)
        pass  # stage A has promotion_allowed=true in YAML for bootstrap purposes


# ─────────────────────────────────────────────────────────────────────────────
# Anti-dependency rules
# ─────────────────────────────────────────────────────────────────────────────


class TestAntiDependencyRules:
    def test_stub_no_promotion_via_stage_decision(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        m = _metrics(win_rate=0.90)
        sm = _s_metrics()
        result = decide_stage_transition(m, sm, stub)
        # stub.promotion_allowed is False
        assert stub.promotion_allowed is False
        assert result == StageDecision.NO_PROMOTION_STUB_OR_DECAY

    def test_decay_no_promotion_via_stage_decision(self, schedule):
        decay = next(s for s in schedule if s.stage_phase == StagePhase.DECAY)
        m = _metrics(win_rate=0.90)
        sm = _s_metrics()
        result = decide_stage_transition(m, sm, decay)
        assert decay.promotion_allowed is False
        assert result == StageDecision.NO_PROMOTION_STUB_OR_DECAY

    def test_clean_promotion_blocked_if_greedy_mix_nonzero(self, schedule):
        clean = next(s for s in schedule if s.stage_id == "stage_d2_gl_std_map_pool_clean")
        # Create a dirty variant with non-zero greedy_mix
        dirty_clean = StageConfig(
            stage_id=clean.stage_id,
            stage_family=clean.stage_family,
            stage_phase=clean.stage_phase,
            distribution_change=clean.distribution_change,
            promotion_allowed=clean.promotion_allowed,
            map_pool_mode=clean.map_pool_mode,
            map_pool=clean.map_pool,
            co_pool_mode=clean.co_pool_mode,
            co_pool=clean.co_pool,
            tier_mode=clean.tier_mode,
            tier=clean.tier,
            scaffolds=ScaffoldConfig(learner_greedy_mix=0.05, capture_move_gate=False, opening_book_prob=0.0),
            cold_opponent=clean.cold_opponent,
            learner_greedy_mix_schedule=clean.learner_greedy_mix_schedule,
            capture_move_gate_schedule=clean.capture_move_gate_schedule,
            opening_book_prob_schedule=clean.opening_book_prob_schedule,
            min_episodes=clean.min_episodes,
            stub_to_decay=clean.stub_to_decay,
            decay_to_clean=clean.decay_to_clean,
            rollback_criteria=clean.rollback_criteria,
            promotion_criteria=clean.promotion_criteria,
            description=clean.description,
        )
        m = _metrics(win_rate=0.70)
        sm = _s_metrics()
        result = decide_stage_transition(m, sm, dirty_clean)
        assert result == StageDecision.NO_PROMOTION_GREEDY_MIX_ACTIVE

    def test_clean_promotion_blocked_if_capture_gate_on(self, schedule):
        clean = next(s for s in schedule if s.stage_id == "stage_d2_gl_std_map_pool_clean")
        dirty_clean = StageConfig(
            stage_id=clean.stage_id,
            stage_family=clean.stage_family,
            stage_phase=clean.stage_phase,
            distribution_change=clean.distribution_change,
            promotion_allowed=clean.promotion_allowed,
            map_pool_mode=clean.map_pool_mode,
            map_pool=clean.map_pool,
            co_pool_mode=clean.co_pool_mode,
            co_pool=clean.co_pool,
            tier_mode=clean.tier_mode,
            tier=clean.tier,
            scaffolds=ScaffoldConfig(learner_greedy_mix=0.0, capture_move_gate=True, opening_book_prob=0.0),
            cold_opponent=clean.cold_opponent,
            learner_greedy_mix_schedule=clean.learner_greedy_mix_schedule,
            capture_move_gate_schedule=clean.capture_move_gate_schedule,
            opening_book_prob_schedule=clean.opening_book_prob_schedule,
            min_episodes=clean.min_episodes,
            stub_to_decay=clean.stub_to_decay,
            decay_to_clean=clean.decay_to_clean,
            rollback_criteria=clean.rollback_criteria,
            promotion_criteria=clean.promotion_criteria,
            description=clean.description,
        )
        m = _metrics(win_rate=0.70)
        sm = _s_metrics()
        result = decide_stage_transition(m, sm, dirty_clean)
        assert result == StageDecision.NO_PROMOTION_CAPTURE_GATE_ACTIVE

    def test_clean_promotion_blocked_if_opening_book_nonzero(self, schedule):
        clean = next(s for s in schedule if s.stage_id == "stage_d2_gl_std_map_pool_clean")
        dirty_clean = StageConfig(
            stage_id=clean.stage_id,
            stage_family=clean.stage_family,
            stage_phase=clean.stage_phase,
            distribution_change=clean.distribution_change,
            promotion_allowed=clean.promotion_allowed,
            map_pool_mode=clean.map_pool_mode,
            map_pool=clean.map_pool,
            co_pool_mode=clean.co_pool_mode,
            co_pool=clean.co_pool,
            tier_mode=clean.tier_mode,
            tier=clean.tier,
            scaffolds=ScaffoldConfig(learner_greedy_mix=0.0, capture_move_gate=False, opening_book_prob=0.25),
            cold_opponent=clean.cold_opponent,
            learner_greedy_mix_schedule=clean.learner_greedy_mix_schedule,
            capture_move_gate_schedule=clean.capture_move_gate_schedule,
            opening_book_prob_schedule=clean.opening_book_prob_schedule,
            min_episodes=clean.min_episodes,
            stub_to_decay=clean.stub_to_decay,
            decay_to_clean=clean.decay_to_clean,
            rollback_criteria=clean.rollback_criteria,
            promotion_criteria=clean.promotion_criteria,
            description=clean.description,
        )
        m = _metrics(win_rate=0.70)
        sm = _s_metrics()
        result = decide_stage_transition(m, sm, dirty_clean)
        assert result == StageDecision.NO_PROMOTION_OPENING_BOOK_ACTIVE


# ─────────────────────────────────────────────────────────────────────────────
# Stage lookup and next-stage resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestFindNextStage:
    def test_within_family_stub_to_decay(self, schedule):
        stub = next(s for s in schedule if s.stage_id == "stage_d0_gl_std_map_pool_stub")
        nxt = _find_next_stage(schedule, stub)
        assert nxt is not None
        assert nxt.stage_id == "stage_d1_gl_std_map_pool_decay"

    def test_within_family_decay_to_clean(self, schedule):
        decay = next(s for s in schedule if s.stage_id == "stage_d1_gl_std_map_pool_decay")
        nxt = _find_next_stage(schedule, decay)
        assert nxt is not None
        assert nxt.stage_id == "stage_d2_gl_std_map_pool_clean"

    def test_clean_is_last_of_family(self, schedule):
        clean = next(s for s in schedule if s.stage_id == "stage_d2_gl_std_map_pool_clean")
        nxt = _find_next_stage(schedule, clean)
        # clean is last in D family; should advance to next family or None
        assert nxt is None or nxt.stage_family > clean.stage_family


# ─────────────────────────────────────────────────────────────────────────────
# CurriculumState read/write
# ─────────────────────────────────────────────────────────────────────────────


class TestCurriculumStateIO:
    def test_write_and_read_roundtrip(self, tmp_path: Path):
        state = CurriculumState(
            current_stage_name="stage_d0_gl_std_map_pool_stub",
            games_observed_in_stage=347,
            entered_stage_at_ts=1700000000.0,
            last_proposal_ts=1700003600.0,
            last_seen_finished_games=1347,
            decay_phase=0,
        )
        path = tmp_path / "curriculum_state.json"
        write_state(path, state)
        loaded = read_state(path)
        assert loaded.current_stage_name == "stage_d0_gl_std_map_pool_stub"
        assert loaded.games_observed_in_stage == 347
        assert loaded.decay_phase == 0

    def test_read_missing_file_returns_default(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        state = read_state(path)
        # Default stage is stage_a_capture_bootstrap
        assert "stage" in state.current_stage_name
        assert state.games_observed_in_stage == 0


# ─────────────────────────────────────────────────────────────────────────────
# compute_proposal integration
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeProposalStable:
    def test_proposal_contains_scaffold_metrics(self, minimal_yaml: Path, tmp_path: Path):
        # Create a dummy game log with no rows
        log_path = tmp_path / "game_log.jsonl"
        log_path.write_text("", encoding="utf-8")

        schedule = load_stages_yaml(minimal_yaml)
        state = CurriculumState(
            current_stage_name="stage_d0_gl_std_map_pool_stub",
            games_observed_in_stage=0,
            entered_stage_at_ts=0.0,
            last_proposal_ts=0.0,
            last_seen_finished_games=0,
            decay_phase=0,
        )
        prop, new_state = compute_proposal_stable(
            game_logs_path=log_path,
            prev_state=state,
            schedule=schedule,
            window_games=200,
        )
        # Should have scaffold metrics even with 0 games
        assert hasattr(prop, "scaffold_metrics")
        assert hasattr(prop, "decision")
        assert prop.decision == "STAY"

    def test_proposal_args_have_required_keys(self, minimal_yaml: Path, tmp_path: Path):
        log_path = tmp_path / "game_log.jsonl"
        log_path.write_text("", encoding="utf-8")

        schedule = load_stages_yaml(minimal_yaml)
        state = CurriculumState(
            current_stage_name="stage_a_capture_bootstrap",
            games_observed_in_stage=0,
            entered_stage_at_ts=0.0,
            last_proposal_ts=0.0,
            last_seen_finished_games=0,
            decay_phase=0,
        )
        prop, _ = compute_proposal_stable(
            game_logs_path=log_path,
            prev_state=state,
            schedule=schedule,
        )
        overrides = prop.args_overrides
        # Must contain scaffold keys
        assert "--learner-greedy-mix" in overrides
        assert "--capture-move-gate" in overrides
        assert "--opening-book-prob" in overrides
        assert "--cold-opponent" in overrides
        assert "--curriculum-tag" in overrides


# ─────────────────────────────────────────────────────────────────────────────
# is_scaffold_free and is_widening helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestStageConfigHelpers:
    def test_is_scaffold_free_true_for_clean(self, schedule):
        clean = next(s for s in schedule if s.stage_id == "stage_d2_gl_std_map_pool_clean")
        assert clean.is_scaffold_free() is True

    def test_is_scaffold_free_false_for_stub(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        assert stub.is_scaffold_free() is False

    def test_is_widening_true_for_stub(self, schedule):
        stub = next(s for s in schedule if s.stage_phase == StagePhase.STUB)
        assert stub.is_widening() is True

    def test_is_widening_false_for_clean(self, schedule):
        clean = next(s for s in schedule if s.stage_phase == StagePhase.CLEAN and s.promotion_allowed)
        assert clean.is_widening() is False


# ─────────────────────────────────────────────────────────────────────────────
# Mock game_log parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMetricsFromLog:
    def test_empty_log_returns_zeros(self, tmp_path: Path):
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("", encoding="utf-8")
        m, sm = compute_metrics(log_path, window_games=200)
        assert m.games_in_window == 0
        assert m.win_rate == 0.0
        assert sm.teacher_override_rate == 0.0
        assert sm.capture_gate_intervention_rate == 0.0

    def test_single_row_parsed(self, tmp_path: Path):
        log_path = tmp_path / "one_game.jsonl"
        row = {
            "turns": 25,
            "done": True,
            "winner": 0,
            "learner_win": True,
            "captures_completed_p0": 5,
            "first_p0_capture_p0_step": 7,
            "terrain_usage_p0": 0.55,
            "losses_hp": [10, 25],
            "p0_env_steps": 500,
            "learner_teacher_overrides": 20,
            "capture_gate_interventions": 15,
            "invalid_action_count": 0,
        }
        log_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        m, sm = compute_metrics(log_path, window_games=200)
        assert m.games_in_window == 1
        assert m.win_rate == 1.0
        assert m.median_captures_completed_p0 == 5.0
        assert m.median_first_p0_capture_step == 7.0
        assert sm.teacher_override_rate == pytest.approx(20 / 500)
        assert sm.capture_gate_intervention_rate == pytest.approx(15 / 500)
        assert sm.terminal_rate == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Decision enum helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestStageDecisionHelpers:
    def test_can_promote_only_when_promote(self):
        assert StageDecision.PROMOTE.is_reject() is False
        assert StageDecision.FAIL_WINRATE.is_reject() is True
        assert StageDecision.INVALID_EVAL_TRUNCATION.is_reject() is True
        assert StageDecision.NO_PROMOTION_STUB_OR_DECAY.is_reject() is True
