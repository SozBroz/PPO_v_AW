# -*- coding: utf-8 -*-
"""Curriculum advisor: orchestrator-owned, auto-tunes train.py args based on observed competence.

The advisor reads game_log.jsonl, classifies the policy into a curriculum stage,
and emits a CurriculumProposal that the orchestrator merges into proposed_args.json.
The operator never has to edit proposed_args.json by hand.

**Sub-stage architecture:** whenever the CO pool or map pool widens, the curriculum
creates a scaffolded sub-stage sequence:

    stub -> decay -> clean -> promotion_eval

- **stub**: widened distribution + heavy scaffolds. No promotion.
- **decay**: scaffolds decaying on schedule. No promotion.
- **clean**: scaffolds at zero. Promotion allowed if criteria pass.
- **promotion_eval**: clean eval gate before advancing to the next major stage.

Promotion is only allowed when:
    stage_phase in (CLEAN, PROMOTION_EVAL)
    learner_greedy_mix == 0.0
    capture_move_gate == False
    opening_book_prob == 0.0

``curriculum_state.json`` uses ``current_stage_name`` (e.g. ``"stage_d0_gl_std_map_pool_stub"``).
The shorthand ``"stage_d"`` maps to the first sub-stage of family D.
See :func:`normalize_curriculum_stage_name`.

Stages are loaded from ``data/curriculum/stages.yaml``. The file defines the full
sub-stage sequence for each major curriculum stage family.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator, TypeAlias

import yaml

from rl.game_log_win import game_log_row_learner_win

FLAG_PRESENT = "_FLAG_PRESENT"

PROBE_OWNED_KEYS = frozenset({"--n-envs", "--n-steps", "--batch-size"})

# Opening book: std_pool_precombat.jsonl (seat 0 + seat 1 lines per map/session).
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENING_BOOK_TRAIN_ARGS: dict[str, Any] = {
    "--opening-book": str(_REPO_ROOT / "data" / "opening_books" / "std_pool_precombat.jsonl"),
    "--opening-book-seats": "both",
    "--opening-book-prob": 1.0,
}

# Human / hand-edited curriculum_state.json often uses "stage_d"; schedule uses long names.
_CURRICULUM_STAGE_SHORTHAND: dict[str, str] = {
    "stage_a": "stage_a_capture_bootstrap",
    "stage_b": "stage_b_capture_competent",
    "stage_c": "stage_c_terrain_competent",
    "stage_d": "stage_d0_gl_std_map_pool_stub",
    "stage_e": "stage_e0_gl_mixed_co_stub",
    "stage_f": "stage_f0_full_random_stub",
    "stage_g": "stage_g_mcts_eval_ready",
}

# Reverse map: family letter -> first sub-stage name (used for shorthand fallback)
_FAMILY_TO_FIRST_STUB: dict[str, str] = {
    "A": "stage_a_capture_bootstrap",
    "B": "stage_b_capture_competent",
    "C": "stage_c_terrain_competent",
    "D": "stage_d0_gl_std_map_pool_stub",
    "E": "stage_e0_gl_mixed_co_stub",
    "F": "stage_f0_full_random_stub",
    "G": "stage_g_mcts_eval_ready",
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage phase enum
# ─────────────────────────────────────────────────────────────────────────────


class StagePhase(Enum):
    """Sub-stage phase labels."""

    STUB = auto()       # widened distribution + scaffolds active; no promotion
    DECAY = auto()      # scaffolds decaying on schedule; no promotion
    CLEAN = auto()      # scaffolds at zero; promotion allowed if criteria pass
    PROMOTION_EVAL = auto()  # clean eval gate; may advance to next major family

    def __str__(self) -> str:
        return self.name.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Decision enums
# ─────────────────────────────────────────────────────────────────────────────


class SubStageDecision(Enum):
    """decide_substage_transition() return type."""

    STAY = auto()
    ADVANCE_TO_DECAY = auto()
    ADVANCE_TO_CLEAN = auto()
    ROLLBACK = auto()


class StageDecision(Enum):
    """decide_stage_transition() return type."""

    STAY = auto()
    NO_PROMOTION_STUB_OR_DECAY = auto()
    NO_PROMOTION_GREEDY_MIX_ACTIVE = auto()
    NO_PROMOTION_CAPTURE_GATE_ACTIVE = auto()
    NO_PROMOTION_OPENING_BOOK_ACTIVE = auto()
    INVALID_EVAL_TERMINAL_RATE_LOW = auto()
    INVALID_EVAL_TRUNCATION = auto()
    FAIL_WINRATE = auto()
    FAIL_OPENING_CAPTURE_DELAY = auto()
    FAIL_OPENING_CAPTURE_COUNT = auto()
    FAIL_MAP_BUCKET = auto()
    FAIL_CO_BUCKET = auto()
    PROMOTE = auto()

    def is_reject(self) -> bool:
        return self != StageDecision.PROMOTE


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold and decay schedule configs (parsed from YAML)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScaffoldConfig:
    """Scaffold settings for a stage or decay phase."""

    learner_greedy_mix: float = 0.0
    capture_move_gate: bool = False
    opening_book_prob: float = 0.0


@dataclass(frozen=True)
class SubStageCriteria:
    """Numeric criteria for stub-to-decay or decay-to-clean transitions."""

    min_episodes: int = 0
    min_terminal_rate: float = 0.0
    max_max_env_steps_truncation_rate: float = 1.0
    max_invalid_action_rate: float = 1.0
    max_first_capture_step_p50: float = float("inf")
    max_first_capture_step_p90: float = float("inf")
    min_captures_by_day5_p50: float = 0.0
    min_income_by_day5_ratio_vs_book_or_greedy: float = 0.0
    min_income_by_day5_ratio_vs_stub: float = 0.0
    max_teacher_override_rate: float = 1.0
    max_capture_gate_intervention_rate: float = 1.0
    min_clean_probe_capture_sanity_rate: float = 0.0
    min_clean_probe_terminal_rate: float = 0.0
    max_clean_probe_truncation_rate: float = 1.0
    clean_probe_wr_not_worse_than_stub_by_more_than: float = 1.0
    learner_greedy_mix_at_end: float = 0.0
    capture_move_gate_at_end: bool = False
    opening_book_prob_at_end: float = 0.0
    max_teacher_override_rate_final: float = 1.0
    max_capture_gate_intervention_rate_final: float = 1.0


@dataclass(frozen=True)
class RollbackCriteria:
    """Conditions that trigger automatic rollback from a sub-stage."""

    after_episodes: int = 0
    max_invalid_action_rate_above: float | None = None
    first_capture_step_p50_above: float | None = None
    captures_by_day5_p50_below: float | None = None
    terminal_rate_below: float | None = None
    max_env_steps_truncation_rate_above: float | None = None
    clean_probe_capture_sanity_rate_below: float | None = None


@dataclass(frozen=True)
class PromotionCriteria:
    """Gates for clean-stage promotion / next-stage advancement."""

    min_eval_games: int = 100
    preferred_eval_games: int = 200
    min_terminal_rate: float = 0.70
    max_max_env_steps_truncation_rate: float = 0.20
    max_invalid_action_rate: float = 0.001
    min_winrate: float = 0.52
    max_first_capture_step_p50: float = float("inf")
    min_captures_by_day5_p50: float = 0.0
    min_income_by_day5_ratio_vs_book_or_greedy: float = 0.0
    no_single_map_below_wr: float | None = None
    no_single_co_matchup_below_wr: float | None = None


@dataclass(slots=True)
class StageConfig:
    """Fully resolved stage configuration from the YAML catalog."""

    stage_id: str
    stage_family: str
    stage_phase: StagePhase
    distribution_change: str  # "none" | "co_pool" | "map_pool" | "both"
    promotion_allowed: bool

    # Map / CO / tier pool spec
    map_pool_mode: str       # "fixed" | "all_gl_std" | "all_enabled"
    map_pool: list[int] | None
    co_pool_mode: str        # "fixed" | "expanded" | "all_enabled"
    co_pool: list[int] | None
    tier_mode: str | None    # "all_enabled" or None
    tier: int | None

    # Scaffold settings
    scaffolds: ScaffoldConfig
    cold_opponent: str

    # Decay schedules (list of values, one per phase); None = no schedule
    learner_greedy_mix_schedule: list[float] | None = None
    capture_move_gate_schedule: list[bool] | None = None
    opening_book_prob_schedule: list[float] | None = None

    # How many episodes to spend in this sub-stage before checking gates
    min_episodes: int = 0

    # Sub-stage transition criteria
    stub_to_decay: SubStageCriteria | None = None
    decay_to_clean: SubStageCriteria | None = None

    # Rollback triggers
    rollback_criteria: RollbackCriteria | None = None

    # Promotion gates (only used when promotion_allowed=True)
    promotion_criteria: PromotionCriteria | None = None

    # Human-readable description
    description: str = ""

    def args_overrides(self) -> dict[str, Any]:
        """Build the train.py arg dict from this stage config."""
        overrides: dict[str, Any] = {
            "--learner-greedy-mix": self.scaffolds.learner_greedy_mix,
            "--capture-move-gate": FLAG_PRESENT if self.scaffolds.capture_move_gate else False,
            "--opening-book-prob": self.scaffolds.opening_book_prob,
            "--cold-opponent": self.cold_opponent,
            "--curriculum-tag": self.stage_id,
        }

        # Map pool
        if self.map_pool_mode == "fixed" and self.map_pool:
            overrides["--map-id"] = self.map_pool[0]
        elif self.map_pool_mode in ("all_gl_std", "all_enabled"):
            overrides["--map-id"] = None
        else:
            overrides["--map-id"] = None

        # CO pool
        if self.co_pool_mode == "fixed" and self.co_pool:
            overrides["--co-p0"] = self.co_pool[0]
            overrides["--co-p1"] = self.co_pool[0]
        else:
            overrides["--co-p0"] = None
            overrides["--co-p1"] = None

        # Tier
        if self.tier is not None:
            overrides["--tier"] = self.tier
        elif self.tier_mode == "all_enabled":
            overrides["--tier"] = None

        # Broad
        if self.co_pool_mode in ("all_enabled", "expanded") and self.tier_mode == "all_enabled":
            overrides["--curriculum-broad-prob"] = 1.0
        else:
            overrides["--curriculum-broad-prob"] = 0.0

        # Opening book defaults
        overrides["--opening-book"] = str(
            _REPO_ROOT / "data" / "opening_books" / "std_pool_precombat.jsonl"
        )
        overrides["--opening-book-seats"] = "both"

        return overrides

    def scaffold_at_episode(self, episode_idx: int) -> ScaffoldConfig:
        """Return scaffold config at a given episode offset within this sub-stage.

        For decay stages, the schedule is split into equal-length phases.
        episode_idx is the count of episodes played within the current sub-stage.
        """
        if self.stage_phase != StagePhase.DECAY:
            return self.scaffolds

        schedule_len = len(self.learner_greedy_mix_schedule) if self.learner_greedy_mix_schedule else 1
        phase_len = max(1, self.min_episodes // schedule_len)
        phase = min(episode_idx // phase_len, schedule_len - 1)

        return ScaffoldConfig(
            learner_greedy_mix=(
                self.learner_greedy_mix_schedule[phase]
                if self.learner_greedy_mix_schedule
                else self.scaffolds.learner_greedy_mix
            ),
            capture_move_gate=(
                self.capture_move_gate_schedule[phase]
                if self.capture_move_gate_schedule
                else self.scaffolds.capture_move_gate
            ),
            opening_book_prob=(
                self.opening_book_prob_schedule[phase]
                if self.opening_book_prob_schedule
                else self.scaffolds.opening_book_prob
            ),
        )

    def is_widening(self) -> bool:
        return self.distribution_change != "none"

    def is_scaffold_free(self) -> bool:
        sc = self.scaffolds
        return (
            sc.learner_greedy_mix == 0.0
            and sc.capture_move_gate is False
            and sc.opening_book_prob == 0.0
        )


# ─────────────────────────────────────────────────────────────────────────────
# YAML loader
# ─────────────────────────────────────────────────────────────────────────────


def _parse_phase(value: str) -> StagePhase:
    mapping = {
        "stub": StagePhase.STUB,
        "decay": StagePhase.DECAY,
        "clean": StagePhase.CLEAN,
        "promotion_eval": StagePhase.PROMOTION_EVAL,
    }
    return mapping[value.lower()]


def _parse_criteria(d: dict[str, Any] | None) -> SubStageCriteria | None:
    if d is None:
        return None
    return SubStageCriteria(
        min_episodes=int(d.get("min_episodes", 0)),
        min_terminal_rate=float(d.get("min_terminal_rate", 0.0)),
        max_max_env_steps_truncation_rate=float(d.get("max_max_env_steps_truncation_rate", 1.0)),
        max_invalid_action_rate=float(d.get("max_invalid_action_rate", 1.0)),
        max_first_capture_step_p50=float(d.get("max_first_capture_step_p50", float("inf"))),
        max_first_capture_step_p90=float(d.get("max_first_capture_step_p90", float("inf"))),
        min_captures_by_day5_p50=float(d.get("min_captures_by_day5_p50", 0.0)),
        min_income_by_day5_ratio_vs_book_or_greedy=float(d.get("min_income_by_day5_ratio_vs_book_or_greedy", 0.0)),
        min_income_by_day5_ratio_vs_stub=float(d.get("min_income_by_day5_ratio_vs_stub", 0.0)),
        max_teacher_override_rate=float(d.get("max_teacher_override_rate", 1.0)),
        max_capture_gate_intervention_rate=float(d.get("max_capture_gate_intervention_rate", 1.0)),
        min_clean_probe_capture_sanity_rate=float(d.get("min_clean_probe_capture_sanity_rate", 0.0)),
        min_clean_probe_terminal_rate=float(d.get("min_clean_probe_terminal_rate", 0.0)),
        max_clean_probe_truncation_rate=float(d.get("max_clean_probe_truncation_rate", 1.0)),
        clean_probe_wr_not_worse_than_stub_by_more_than=float(
            d.get("clean_probe_wr_not_worse_than_stub_by_more_than", 1.0)
        ),
        learner_greedy_mix_at_end=float(d.get("learner_greedy_mix_at_end", 0.0)),
        capture_move_gate_at_end=bool(d.get("capture_move_gate_at_end", False)),
        opening_book_prob_at_end=float(d.get("opening_book_prob_at_end", 0.0)),
        max_teacher_override_rate_final=float(d.get("max_teacher_override_rate_final", 1.0)),
        max_capture_gate_intervention_rate_final=float(d.get("max_capture_gate_intervention_rate_final", 1.0)),
    )


def _parse_rollback(d: dict[str, Any] | None) -> RollbackCriteria | None:
    if d is None:
        return None
    return RollbackCriteria(
        after_episodes=int(d.get("after_episodes", 0)),
        max_invalid_action_rate_above=d.get("max_invalid_action_rate_above"),
        first_capture_step_p50_above=d.get("first_capture_step_p50_above"),
        captures_by_day5_p50_below=d.get("captures_by_day5_p50_below"),
        terminal_rate_below=d.get("terminal_rate_below"),
        max_env_steps_truncation_rate_above=d.get("max_env_steps_truncation_rate_above"),
        clean_probe_capture_sanity_rate_below=d.get("clean_probe_capture_sanity_rate_below"),
    )


def _parse_promotion(d: dict[str, Any] | None) -> PromotionCriteria | None:
    if d is None:
        return None
    return PromotionCriteria(
        min_eval_games=int(d.get("min_eval_games", 100)),
        preferred_eval_games=int(d.get("preferred_eval_games", 200)),
        min_terminal_rate=float(d.get("min_terminal_rate", 0.70)),
        max_max_env_steps_truncation_rate=float(d.get("max_max_env_steps_truncation_rate", 0.20)),
        max_invalid_action_rate=float(d.get("max_invalid_action_rate", 0.001)),
        min_winrate=float(d.get("min_winrate", 0.52)),
        max_first_capture_step_p50=float(d.get("max_first_capture_step_p50", float("inf"))),
        min_captures_by_day5_p50=float(d.get("min_captures_by_day5_p50", 0.0)),
        min_income_by_day5_ratio_vs_book_or_greedy=float(
            d.get("min_income_by_day5_ratio_vs_book_or_greedy", 0.0)
        ),
        no_single_map_below_wr=d.get("no_single_map_below_wr"),
        no_single_co_matchup_below_wr=d.get("no_single_co_matchup_below_wr"),
    )


def _parse_scaffolds(d: dict[str, Any]) -> ScaffoldConfig:
    return ScaffoldConfig(
        learner_greedy_mix=float(d.get("learner_greedy_mix", 0.0)),
        capture_move_gate=bool(d.get("capture_move_gate", False)),
        opening_book_prob=float(d.get("opening_book_prob", 0.0)),
    )


def load_stages_yaml(path: Path) -> list[StageConfig]:
    """Load and parse data/curriculum/stages.yaml into StageConfig objects."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    stages_raw = raw.get("stages", [])
    configs: list[StageConfig] = []
    for s in stages_raw:
        scaffold_d = s.get("scaffolds", {})
        cfg = StageConfig(
            stage_id=str(s["stage_id"]),
            stage_family=str(s["stage_family"]),
            stage_phase=_parse_phase(str(s["stage_phase"])),
            distribution_change=str(s.get("distribution_change", "none")),
            promotion_allowed=bool(s.get("promotion_allowed", False)),
            map_pool_mode=str(s.get("map_pool_mode", "fixed")),
            map_pool=s.get("map_pool"),
            co_pool_mode=str(s.get("co_pool_mode", "fixed")),
            co_pool=s.get("co_pool"),
            tier_mode=s.get("tier_mode"),
            tier=s.get("tier"),
            scaffolds=_parse_scaffolds(scaffold_d) if scaffold_d else ScaffoldConfig(),
            cold_opponent=str(s.get("cold_opponent", "random")),
            learner_greedy_mix_schedule=s.get("learner_greedy_mix_schedule"),
            capture_move_gate_schedule=s.get("capture_move_gate_schedule"),
            opening_book_prob_schedule=s.get("opening_book_prob_schedule"),
            min_episodes=int(s.get("min_episodes", 0)),
            stub_to_decay=_parse_criteria(s.get("stub_to_decay")),
            decay_to_clean=_parse_criteria(s.get("decay_to_clean")),
            rollback_criteria=_parse_rollback(s.get("rollback_criteria")),
            promotion_criteria=_parse_promotion(s.get("promotion_criteria")),
            description=str(s.get("description", "")),
        )
        configs.append(cfg)
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def normalize_curriculum_stage_name(name: str) -> str:
    """Map shorthand or legacy names to canonical stage_id strings."""
    s = str(name).strip()
    # Direct shorthand map
    if s in _CURRICULUM_STAGE_SHORTHAND:
        return _CURRICULUM_STAGE_SHORTHAND[s]
    # Family shorthand "stage_d" -> first stub of that family
    if s.startswith("stage_") and len(s) == 7 and s[-1].isalpha():
        family = s[-1].upper()
        if family in _FAMILY_TO_FIRST_STUB:
            return _FAMILY_TO_FIRST_STUB[family]
    # Legacy renames
    if s == "stage_d_self_play_pure":
        return "stage_f_self_play_pure"
    if s == "stage_d_gl_std_map_pool_t3":
        return "stage_d0_gl_std_map_pool_stub"
    if s == "stage_d_gl_std_map_pool_t4":
        return "stage_d0_gl_std_map_pool_stub"
    if s == "stage_e_gl_mixed_ladder":
        return "stage_e0_gl_mixed_co_stub"
    if s == "stage_f_self_play_pure":
        return "stage_f0_full_random_stub"
    return s


def _family_letter(stage_id: str) -> str:
    """Extract family letter from stage_id (e.g. 'D' from 'stage_d0_gl_std_map_pool_stub')."""
    parts = stage_id.split("_")
    if not parts:
        return "A"
    return parts[0].replace("stage_", "").upper()


@dataclass(slots=True)
class CompetenceMetrics:
    """Rolling-window competence (aligned with :class:`MctsHealthMetrics` names where noted)."""

    games_in_window: int
    capture_sense_score: float
    terrain_usage_score: float
    army_value_lead: float
    win_rate: float
    episode_quality: float
    # From ``first_p0_capture_p0_step`` / ``captures_completed_p0`` in game_log (§10g capture gates).
    median_first_p0_capture_step: float = float("inf")
    median_captures_completed_p0: float = 0.0


@dataclass(slots=True)
class ScaffoldMetrics:
    """Scaffold usage rates computed from game_log rows."""

    teacher_override_rate: float = 0.0      # learner_teacher_overrides / max(1, p0_env_steps)
    capture_gate_intervention_rate: float = 0.0  # capture_gate_interventions / max(1, p0_env_steps)
    opening_book_desync_rate: float = 0.0
    opening_book_used_count: int = 0
    opening_book_desync_count: int = 0
    invalid_action_rate: float = 0.0
    terminal_rate: float = 0.0
    max_env_steps_truncation_rate: float = 0.0
    # Clean probe results (populated by tools/run_clean_probe.py reading its output)
    clean_probe_capture_sanity_rate: float | None = None
    clean_probe_terminal_rate: float | None = None
    clean_probe_truncation_rate: float | None = None
    clean_probe_winrate: float | None = None


@dataclass(slots=True)
class CurriculumState:
    current_stage_name: str
    games_observed_in_stage: int
    entered_stage_at_ts: float
    last_proposal_ts: float
    last_seen_finished_games: int = 0
    # Decay phase tracking: which phase of the schedule we are in
    decay_phase: int = 0


@dataclass(slots=True)
class CurriculumProposal:
    stage_name: str
    args_overrides: dict[str, Any]
    reason: str
    metrics_snapshot: CompetenceMetrics
    scaffold_metrics: ScaffoldMetrics
    decision: str  # StageDecision or SubStageDecision name
    promotion_allowed: bool


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_log_lines(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _finished_rows_for_machine(
    path: Path, machine_id: str | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _parse_log_lines(path):
        if "turns" not in row:
            continue
        if machine_id is not None:
            mid = row.get("machine_id")
            if mid is None or str(mid) != str(machine_id):
                continue
        out.append(row)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median_float(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return 0.5 * (s[mid - 1] + s[mid])


def _episode_quality_score(avg_turns: float, early_resign_rate: float) -> float:
    """Scalar 0..1 summarizing episode length + early-quit pressure (MCTS health-style)."""
    if avg_turns >= 25.0 and early_resign_rate <= 0.3:
        return 1.0
    len_part = max(0.0, min(1.0, avg_turns / 25.0))
    early_part = max(0.0, 1.0 - early_resign_rate)
    return max(0.0, min(1.0, 0.5 * len_part + 0.5 * early_part))


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def compute_metrics(
    game_logs_path: Path, window_games: int = 200, machine_id: str | None = None
) -> tuple[CompetenceMetrics, ScaffoldMetrics]:
    rows = _finished_rows_for_machine(Path(game_logs_path), machine_id)
    if len(rows) > int(window_games):
        rows = rows[-int(window_games):]
    n = len(rows)
    if n == 0:
        empty_c = CompetenceMetrics(
            games_in_window=0,
            capture_sense_score=0.0,
            terrain_usage_score=0.0,
            army_value_lead=0.0,
            win_rate=0.0,
            episode_quality=0.0,
            median_first_p0_capture_step=float("inf"),
            median_captures_completed_p0=0.0,
        )
        empty_s = ScaffoldMetrics()
        return empty_c, empty_s

    cap_scores: list[float] = []
    cap_counts: list[float] = []
    first_p0_steps: list[float] = []
    terr: list[float] = []
    army_pos: list[float] = []
    wins: list[float] = []
    turns: list[float] = []
    early: list[float] = []

    # Scaffold numerators / denominators
    teacher_overrides_total = 0
    capture_gate_interventions_total = 0
    p0_steps_total = 0
    invalid_actions_total = 0
    terminated_count = 0
    max_env_steps_count = 0
    opening_book_used = 0
    opening_book_desync = 0

    for r in rows:
        c = r.get("captures_completed_p0")
        c_f = _safe_float(c)
        cap_scores.append(min(1.0, c_f / 3.0))
        cap_counts.append(c_f)

        fcp = r.get("first_p0_capture_p0_step")
        if fcp is not None:
            try:
                first_p0_steps.append(float(fcp))
            except (TypeError, ValueError):
                pass

        tu = r.get("terrain_usage_p0")
        terr.append(_safe_float(tu))

        lh = r.get("losses_hp")
        pos = 0.0
        if isinstance(lh, (list, tuple)) and len(lh) >= 2:
            try:
                p0l, p1l = float(lh[0]), float(lh[1])
                pos = 1.0 if p1l > p0l else 0.0
            except (TypeError, ValueError):
                pos = 0.0
        army_pos.append(pos)

        wins.append(1.0 if game_log_row_learner_win(r) else 0.0)

        t = r.get("turns")
        t_f = _safe_float(t)
        turns.append(t_f)
        early.append(1.0 if t_f < 20.0 else 0.0)

        # Scaffold rates
        p0_steps = r.get("p0_env_steps", 0)
        p0_steps_total += int(p0_steps) if p0_steps else 0

        teacher_overrides = r.get("learner_teacher_overrides", 0)
        teacher_overrides_total += int(teacher_overrides) if teacher_overrides else 0

        gate_int = r.get("capture_gate_interventions", 0)
        capture_gate_interventions_total += int(gate_int) if gate_int else 0

        inv = r.get("invalid_action_count", 0)
        invalid_actions_total += int(inv) if inv else 0

        terminated_count += 1 if r.get("done") else 0

        trunc_reason = r.get("truncation_reason", "")
        if trunc_reason == "max_env_steps":
            max_env_steps_count += 1

        ob_used = r.get("opening_book_used_p0") or r.get("opening_book_used_p1")
        if ob_used:
            opening_book_used += 1
            desync = r.get("opening_book_desync_p0") or r.get("opening_book_desync_p1")
            if desync:
                opening_book_desync += 1

    m_turns = _mean(turns)
    m_early = _mean(early)
    med_cap = _median_float(cap_counts)
    med_fc = _median_float(first_p0_steps)

    denom = max(1, p0_steps_total)
    competence = CompetenceMetrics(
        games_in_window=n,
        capture_sense_score=_mean(cap_scores),
        terrain_usage_score=_mean(terr),
        army_value_lead=_mean(army_pos),
        win_rate=_mean(wins),
        episode_quality=_episode_quality_score(m_turns, m_early),
        median_first_p0_capture_step=float("inf") if med_fc is None else float(med_fc),
        median_captures_completed_p0=float(med_cap) if med_cap is not None else 0.0,
    )
    scaffold = ScaffoldMetrics(
        teacher_override_rate=teacher_overrides_total / denom,
        capture_gate_intervention_rate=capture_gate_interventions_total / denom,
        opening_book_desync_rate=opening_book_desync / max(1, opening_book_used),
        opening_book_used_count=opening_book_used,
        opening_book_desync_count=opening_book_desync,
        invalid_action_rate=invalid_actions_total / denom,
        terminal_rate=terminated_count / n,
        max_env_steps_truncation_rate=max_env_steps_count / n,
    )
    return competence, scaffold


# ─────────────────────────────────────────────────────────────────────────────
# Transition decision logic
# ─────────────────────────────────────────────────────────────────────────────


def _check_criteria(metrics: CompetenceMetrics, s_metrics: ScaffoldMetrics, crit: SubStageCriteria) -> tuple[bool, str]:
    """Return (passed, reason)."""
    if metrics.games_in_window < crit.min_episodes:
        return False, f"games_in_window {metrics.games_in_window} < {crit.min_episodes}"

    if s_metrics.invalid_action_rate > crit.max_invalid_action_rate:
        return False, f"invalid_action_rate {s_metrics.invalid_action_rate:.4f} > {crit.max_invalid_action_rate}"

    if metrics.median_first_p0_capture_step > crit.max_first_capture_step_p50:
        return False, f"first_capture_step_p50 {metrics.median_first_p0_capture_step} > {crit.max_first_capture_step_p50}"

    if metrics.median_captures_completed_p0 < crit.min_captures_by_day5_p50:
        return False, f"captures_by_day5_p50 {metrics.median_captures_completed_p0} < {crit.min_captures_by_day5_p50}"

    if s_metrics.teacher_override_rate > crit.max_teacher_override_rate:
        return False, f"teacher_override_rate {s_metrics.teacher_override_rate:.4f} > {crit.max_teacher_override_rate}"

    if s_metrics.capture_gate_intervention_rate > crit.max_capture_gate_intervention_rate:
        return False, f"capture_gate_intervention_rate {s_metrics.capture_gate_intervention_rate:.4f} > {crit.max_capture_gate_intervention_rate}"

    if s_metrics.clean_probe_capture_sanity_rate is not None and s_metrics.clean_probe_capture_sanity_rate < crit.min_clean_probe_capture_sanity_rate:
        return False, f"clean_probe_capture_sanity_rate {s_metrics.clean_probe_capture_sanity_rate:.4f} < {crit.min_clean_probe_capture_sanity_rate}"

    return True, "passed"


def decide_substage_transition(
    metrics: CompetenceMetrics,
    s_metrics: ScaffoldMetrics,
    stage: StageConfig,
    games_in_stage: int,
) -> SubStageDecision:
    """Determine whether to stay, advance, or rollback from a stub/decay sub-stage."""
    phase = stage.stage_phase

    if phase not in (StagePhase.STUB, StagePhase.DECAY):
        return SubStageDecision.STAY

    # Check rollback first
    rb = stage.rollback_criteria
    if rb is not None and games_in_stage >= rb.after_episodes:
        if rb.max_invalid_action_rate_above is not None and s_metrics.invalid_action_rate > rb.max_invalid_action_rate_above:
            return SubStageDecision.ROLLBACK
        if rb.first_capture_step_p50_above is not None and metrics.median_first_p0_capture_step > rb.first_capture_step_p50_above:
            return SubStageDecision.ROLLBACK
        if rb.captures_by_day5_p50_below is not None and metrics.median_captures_completed_p0 < rb.captures_by_day5_p50_below:
            return SubStageDecision.ROLLBACK
        if rb.terminal_rate_below is not None and s_metrics.terminal_rate < rb.terminal_rate_below:
            return SubStageDecision.ROLLBACK
        if rb.max_env_steps_truncation_rate_above is not None and s_metrics.max_env_steps_truncation_rate > rb.max_env_steps_truncation_rate_above:
            return SubStageDecision.ROLLBACK
        if rb.clean_probe_capture_sanity_rate_below is not None and s_metrics.clean_probe_capture_sanity_rate is not None and s_metrics.clean_probe_capture_sanity_rate < rb.clean_probe_capture_sanity_rate_below:
            return SubStageDecision.ROLLBACK

    if phase == StagePhase.STUB:
        crit = stage.stub_to_decay
        if crit is None:
            return SubStageDecision.STAY
        passed, reason = _check_criteria(metrics, s_metrics, crit)
        if not passed:
            return SubStageDecision.STAY
        return SubStageDecision.ADVANCE_TO_DECAY

    if phase == StagePhase.DECAY:
        crit = stage.decay_to_clean
        if crit is None:
            return SubStageDecision.STAY

        # Additional gate: scaffolds must be at/near zero in final decay phase
        if crit.learner_greedy_mix_at_end > 0.0:
            return SubStageDecision.STAY
        if crit.capture_move_gate_at_end:
            return SubStageDecision.STAY
        if crit.opening_book_prob_at_end > 0.0:
            return SubStageDecision.STAY

        # Check clean probe
        if s_metrics.clean_probe_capture_sanity_rate is not None and s_metrics.clean_probe_capture_sanity_rate < crit.min_clean_probe_capture_sanity_rate:
            return SubStageDecision.STAY

        if s_metrics.clean_probe_terminal_rate is not None and s_metrics.clean_probe_terminal_rate < crit.min_clean_probe_terminal_rate:
            return SubStageDecision.STAY

        if s_metrics.clean_probe_truncation_rate is not None and s_metrics.clean_probe_truncation_rate > crit.max_clean_probe_truncation_rate:
            return SubStageDecision.STAY

        # Winrate regression check
        if s_metrics.clean_probe_winrate is not None and metrics.win_rate > 0:
            wr_drop = metrics.win_rate - s_metrics.clean_probe_winrate
            if wr_drop > crit.clean_probe_wr_not_worse_than_stub_by_more_than:
                return SubStageDecision.STAY

        passed, _ = _check_criteria(metrics, s_metrics, crit)
        if not passed:
            return SubStageDecision.STAY
        return SubStageDecision.ADVANCE_TO_CLEAN

    return SubStageDecision.STAY


def decide_stage_transition(
    metrics: CompetenceMetrics,
    s_metrics: ScaffoldMetrics,
    stage: StageConfig,
) -> StageDecision:
    """Determine whether a CLEAN or PROMOTION_EVAL stage can advance to the next family."""
    if not stage.promotion_allowed:
        return StageDecision.NO_PROMOTION_STUB_OR_DECAY

    sc = stage.scaffolds
    if sc.learner_greedy_mix > 0.0:
        return StageDecision.NO_PROMOTION_GREEDY_MIX_ACTIVE
    if sc.capture_move_gate:
        return StageDecision.NO_PROMOTION_CAPTURE_GATE_ACTIVE
    if sc.opening_book_prob > 0.0:
        return StageDecision.NO_PROMOTION_OPENING_BOOK_ACTIVE

    pc = stage.promotion_criteria
    if pc is None:
        return StageDecision.STAY

    if s_metrics.terminal_rate < pc.min_terminal_rate:
        return StageDecision.INVALID_EVAL_TERMINAL_RATE_LOW

    if s_metrics.max_env_steps_truncation_rate > pc.max_max_env_steps_truncation_rate:
        return StageDecision.INVALID_EVAL_TRUNCATION

    if s_metrics.invalid_action_rate > pc.max_invalid_action_rate:
        return StageDecision.NO_PROMOTION_GREEDY_MIX_ACTIVE  # reuse; no action rate gate yet

    if metrics.win_rate < pc.min_winrate:
        return StageDecision.FAIL_WINRATE

    if metrics.median_first_p0_capture_step > pc.max_first_capture_step_p50:
        return StageDecision.FAIL_OPENING_CAPTURE_DELAY

    if metrics.median_captures_completed_p0 < pc.min_captures_by_day5_p50:
        return StageDecision.FAIL_OPENING_CAPTURE_COUNT

    return StageDecision.PROMOTE


# ─────────────────────────────────────────────────────────────────────────────
# Stage catalog and lookup
# ─────────────────────────────────────────────────────────────────────────────


def _load_default_schedule() -> list[StageConfig]:
    """Load stages.yaml; fall back to a minimal hardcoded schedule if absent."""
    yaml_path = _REPO_ROOT / "data" / "curriculum" / "stages.yaml"
    if yaml_path.is_file():
        return load_stages_yaml(yaml_path)

    # Minimal fallback — covers only stages that existed before this plan
    return [
        StageConfig(
            stage_id="stage_a_capture_bootstrap",
            stage_family="A",
            stage_phase=StagePhase.CLEAN,
            distribution_change="none",
            promotion_allowed=True,
            map_pool_mode="fixed",
            map_pool=[123858],
            co_pool_mode="fixed",
            co_pool=[14],
            scaffolds=ScaffoldConfig(learner_greedy_mix=0.30, capture_move_gate=True, opening_book_prob=1.0),
            cold_opponent="greedy_capture",
            min_episodes=200,
        ),
    ]


DEFAULT_SCHEDULE: list[StageConfig] = _load_default_schedule()


def _stage_index_by_id(schedule: list[StageConfig], stage_id: str) -> int:
    for i, s in enumerate(schedule):
        if s.stage_id == stage_id:
            return i
    return -1


def _find_next_stage(schedule: list[StageConfig], current: StageConfig) -> StageConfig | None:
    """Find the next stage in the schedule.

    For widening families (D, E, F), advance within the family:
      stub -> decay -> clean -> promotion_eval
    For non-widening families (A, B, C), advance to the next family.
    """
    # Within-family advance
    family_stages = [s for s in schedule if s.stage_family == current.stage_family]
    if len(family_stages) > 1:
        for i, s in enumerate(family_stages):
            if s.stage_id == current.stage_id and i + 1 < len(family_stages):
                return family_stages[i + 1]
        # Already at last sub-stage of this family; advance to next family's first stub
        next_family = chr(ord(current.stage_family) + 1)
        for s in schedule:
            if s.stage_family == next_family and s.stage_phase == StagePhase.STUB:
                return s
        # Fallback: next stage in schedule
        idx = _stage_index_by_id(schedule, current.stage_id)
        if idx >= 0 and idx + 1 < len(schedule):
            return schedule[idx + 1]
        return None

    # Non-widening: simple next in list
    idx = _stage_index_by_id(schedule, current.stage_id)
    if idx >= 0 and idx + 1 < len(schedule):
        return schedule[idx + 1]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum state read/write
# ─────────────────────────────────────────────────────────────────────────────


def _default_curriculum_state() -> CurriculumState:
    return CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].stage_id,
        games_observed_in_stage=0,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
        decay_phase=0,
    )


def read_state(path: Path) -> CurriculumState:
    if not path.is_file():
        return _default_curriculum_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_curriculum_state()
    try:
        name = normalize_curriculum_stage_name(str(raw["current_stage_name"]))
        return CurriculumState(
            current_stage_name=name,
            games_observed_in_stage=int(raw["games_observed_in_stage"]),
            entered_stage_at_ts=float(raw["entered_stage_at_ts"]),
            last_proposal_ts=float(raw["last_proposal_ts"]),
            last_seen_finished_games=int(raw.get("last_seen_finished_games", 0)),
            decay_phase=int(raw.get("decay_phase", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return _default_curriculum_state()


def write_state(path: Path, state: CurriculumState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(state), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix="curriculum_state_", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Proposal computation
# ─────────────────────────────────────────────────────────────────────────────


def next_curriculum_state_after_tick(
    *,
    game_logs_path: Path,
    prev: CurriculumState,
    schedule: list[StageConfig] | None = None,
    window_games: int = 200,
    machine_id: str | None = None,
    now_ts: float | None = None,
) -> CurriculumState:
    """Advance counters and possibly stage; used by orchestrator to persist curriculum_state.json."""
    import time as _time

    if schedule is None:
        schedule = DEFAULT_SCHEDULE

    now = float(now_ts if now_ts is not None else _time.time())
    metrics, s_metrics = compute_metrics(game_logs_path, window_games=window_games, machine_id=machine_id)
    total_games = len(_finished_rows_for_machine(Path(game_logs_path), machine_id))
    delta = max(0, total_games - prev.last_seen_finished_games)

    current_idx = _stage_index_by_id(schedule, prev.current_stage_name)
    if current_idx < 0:
        current_idx = 0
    current_stage = schedule[current_idx]

    # Determine transition
    decision: str = "STAY"
    decay_phase = prev.decay_phase

    if current_stage.stage_phase in (StagePhase.STUB, StagePhase.DECAY):
        sub_dec = decide_substage_transition(metrics, s_metrics, current_stage, prev.games_observed_in_stage)
        decision = sub_dec.name

        if sub_dec == SubStageDecision.ADVANCE_TO_DECAY:
            nxt = _find_next_stage(schedule, current_stage)
            if nxt is not None:
                return CurriculumState(
                    current_stage_name=nxt.stage_id,
                    games_observed_in_stage=delta,
                    entered_stage_at_ts=now,
                    last_proposal_ts=now,
                    last_seen_finished_games=total_games,
                    decay_phase=0,
                )
        elif sub_dec == SubStageDecision.ADVANCE_TO_CLEAN:
            nxt = _find_next_stage(schedule, current_stage)
            if nxt is not None:
                return CurriculumState(
                    current_stage_name=nxt.stage_id,
                    games_observed_in_stage=delta,
                    entered_stage_at_ts=now,
                    last_proposal_ts=now,
                    last_seen_finished_games=total_games,
                    decay_phase=0,
                )
        elif sub_dec == SubStageDecision.ROLLBACK:
            # Roll back to the clean stage of the previous family, or first clean stage
            prev_family = chr(ord(current_stage.stage_family) - 1) if current_stage.stage_family > "A" else "A"
            for s in reversed(schedule):
                if s.stage_family == prev_family and s.stage_phase == StagePhase.CLEAN:
                    return CurriculumState(
                        current_stage_name=s.stage_id,
                        games_observed_in_stage=delta,
                        entered_stage_at_ts=now,
                        last_proposal_ts=now,
                        last_seen_finished_games=total_games,
                        decay_phase=0,
                    )
            # Fallback: go to stage A
            for s in schedule:
                if s.stage_phase == StagePhase.CLEAN and s.stage_family == "A":
                    return CurriculumState(
                        current_stage_name=s.stage_id,
                        games_observed_in_stage=delta,
                        entered_stage_at_ts=now,
                        last_proposal_ts=now,
                        last_seen_finished_games=total_games,
                        decay_phase=0,
                    )

    elif current_stage.stage_phase == StagePhase.CLEAN:
        stage_dec = decide_stage_transition(metrics, s_metrics, current_stage)
        decision = stage_dec.name
        if stage_dec == StageDecision.PROMOTE:
            nxt = _find_next_stage(schedule, current_stage)
            if nxt is not None:
                return CurriculumState(
                    current_stage_name=nxt.stage_id,
                    games_observed_in_stage=delta,
                    entered_stage_at_ts=now,
                    last_proposal_ts=now,
                    last_seen_finished_games=total_games,
                    decay_phase=0,
                )

    return CurriculumState(
        current_stage_name=current_stage.stage_id,
        games_observed_in_stage=prev.games_observed_in_stage + delta,
        entered_stage_at_ts=prev.entered_stage_at_ts,
        last_proposal_ts=now,
        last_seen_finished_games=total_games,
        decay_phase=decay_phase,
    )


def compute_proposal_stable(
    game_logs_path: Path,
    prev_state: CurriculumState,
    schedule: list[StageConfig] | None = None,
    *,
    window_games: int = 200,
    machine_id: str | None = None,
    now_ts: float | None = None,
) -> tuple[CurriculumProposal, CurriculumState]:
    """
    Single call path: proposal args match ``next_curriculum_state_after_tick`` stage.
    """
    if schedule is None:
        schedule = DEFAULT_SCHEDULE

    metrics, s_metrics = compute_metrics(game_logs_path, window_games=window_games, machine_id=machine_id)
    total_games = len(_finished_rows_for_machine(Path(game_logs_path), machine_id))

    current_idx = _stage_index_by_id(schedule, prev_state.current_stage_name)
    if current_idx < 0:
        current_idx = 0
    current_stage = schedule[current_idx]

    # Determine decision and reason
    decision_str = "STAY"
    reason = "holding"

    if current_stage.stage_phase in (StagePhase.STUB, StagePhase.DECAY):
        sub_dec = decide_substage_transition(metrics, s_metrics, current_stage, prev_state.games_observed_in_stage)
        decision_str = sub_dec.name
        if sub_dec == SubStageDecision.ADVANCE_TO_DECAY:
            reason = f"advance stub->decay: {current_stage.stage_id}"
        elif sub_dec == SubStageDecision.ADVANCE_TO_CLEAN:
            reason = f"advance decay->clean: {current_stage.stage_id}"
        elif sub_dec == SubStageDecision.ROLLBACK:
            reason = f"rollback from {current_stage.stage_id}"
        else:
            reason = f"holding {current_stage.stage_id}: criteria not met"
    elif current_stage.stage_phase == StagePhase.CLEAN:
        stage_dec = decide_stage_transition(metrics, s_metrics, current_stage)
        decision_str = stage_dec.name
        if stage_dec == StageDecision.PROMOTE:
            reason = f"promote {current_stage.stage_id}"
        else:
            reason = f"stay {current_stage.stage_id}: {stage_dec.name}"

    # Build scaffold-aware overrides
    scaffolds = current_stage.scaffolds
    if current_stage.stage_phase == StagePhase.DECAY and prev_state.decay_phase >= 0:
        sc = current_stage.scaffold_at_episode(prev_state.games_observed_in_stage)
        scaffolds = sc

    overrides = dict(current_stage.args_overrides())
    overrides["--learner-greedy-mix"] = scaffolds.learner_greedy_mix
    overrides["--capture-move-gate"] = FLAG_PRESENT if scaffolds.capture_move_gate else False
    overrides["--opening-book-prob"] = scaffolds.opening_book_prob
    for k in PROBE_OWNED_KEYS:
        overrides.pop(k, None)

    proposal = CurriculumProposal(
        stage_name=current_stage.stage_id,
        args_overrides=overrides,
        reason=reason,
        metrics_snapshot=metrics,
        scaffold_metrics=s_metrics,
        decision=decision_str,
        promotion_allowed=current_stage.promotion_allowed,
    )

    new_state = next_curriculum_state_after_tick(
        game_logs_path=game_logs_path,
        prev=prev_state,
        schedule=schedule,
        window_games=window_games,
        machine_id=machine_id,
        now_ts=now_ts,
    )
    return proposal, new_state


def compute_proposal(
    game_logs_path: Path,
    current_state: CurriculumState,
    schedule: list[StageConfig] | None = None,
    *,
    window_games: int = 200,
    machine_id: str | None = None,
    now_ts: float | None = None,
) -> CurriculumProposal:
    prop, _st = compute_proposal_stable(
        game_logs_path,
        current_state,
        schedule,
        window_games=window_games,
        machine_id=machine_id,
        now_ts=now_ts,
    )
    return prop


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--game-log",
        type=Path,
        default=None,
        help="Path to game_log.jsonl (default: <repo>/logs/game_log.jsonl)",
    )
    ap.add_argument(
        "--machine-id",
        type=str,
        default=None,
        help="Filter finished games to this AWBW_MACHINE_ID (optional)",
    )
    ap.add_argument(
        "--window-games",
        type=int,
        default=200,
        help="Rolling window (default 200 per §10g fleet narrative)",
    )
    ap.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional curriculum_state.json to read/write for multi-call simulation",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    ap.add_argument(
        "--stages-yaml",
        type=Path,
        default=None,
        help="Path to stages.yaml (default: <repo-root>/data/curriculum/stages.yaml)",
    )
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    gpath = args.game_log or (repo / "logs" / "game_log.jsonl")
    yaml_path = args.stages_yaml or (repo / "data" / "curriculum" / "stages.yaml")

    if yaml_path.is_file():
        schedule = load_stages_yaml(yaml_path)
    else:
        schedule = DEFAULT_SCHEDULE

    st: CurriculumState
    if args.state_file is not None:
        st = read_state(Path(args.state_file))
    else:
        st = _default_curriculum_state()

    prop, st_new = compute_proposal_stable(
        gpath,
        st,
        schedule,
        window_games=int(args.window_games),
        machine_id=args.machine_id,
    )
    out = {
        "proposal": {
            "stage_name": prop.stage_name,
            "args_overrides": prop.args_overrides,
            "reason": prop.reason,
            "decision": prop.decision,
            "promotion_allowed": prop.promotion_allowed,
        },
        "competence_metrics": asdict(prop.metrics_snapshot),
        "scaffold_metrics": asdict(prop.scaffold_metrics),
        "updated_state": asdict(st_new),
    }
    print(json.dumps(out, indent=2))
    if args.state_file is not None:
        write_state(Path(args.state_file), st_new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
