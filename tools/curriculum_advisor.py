# -*- coding: utf-8 -*-
"""Curriculum advisor: orchestrator-owned, auto-tunes train.py args based on observed competence.

The advisor reads game_log.jsonl, classifies the policy into a curriculum stage,
and emits a CurriculumProposal that the orchestrator merges into proposed_args.json.
The operator never has to edit proposed_args.json by hand.

``curriculum_state.json`` may use shorthand ``current_stage_name`` values
(``"stage_d"``, …); they are normalized to :data:`DEFAULT_SCHEDULE` names.
Unknown strings still fall through to index 0 (bootstrap) — use
:func:`normalize_curriculum_stage_name` or copy the exact ``name`` from the schedule.

**DRAFT schedule:** DEFAULT_SCHEDULE is an initial ratchet for pool bootstrap; gates
are aligned with `.cursor/plans/train.py_fps_campaign_c26ce6d4.plan.md` §10g
(200-game rolling competence, capture medians, stricter win/terrain/episode-quality
bars per stage). The operator can replace the schedule in code or via a future hook.

Stages **D–E** widen training to Global League **Std** map diversity and then
mixed tier/CO sampling. Use JSON ``null`` for ``--map-id`` / ``--tier`` /
``--co-p0`` / ``--co-p1`` in ``proposed_args.json`` so :func:`scripts.fleet_orchestrator.build_train_argv_from_proposed_args`
omits those flags (``train.py`` defaults: all Std maps in ``gl_map_pool.json``,
random enabled tier and COs). See ``DEFAULT_SCHEDULE`` stage names.

**Broad** means ``--curriculum-broad-prob`` (see :func:`rl.env.sample_training_matchup`):
each reset, with that probability the env ignores the narrow ``tier`` / ``co_p0`` /
``co_p1`` pins and samples a **fully random** matchup — random Std-pool map, random
enabled tier on that map, random COs from that tier. At ``1.0`` every curriculum
episode is fully random (narrow pins unused). **Live PPO workers** (env constructed
with a ``live_snapshot_path``) force broad to ``0`` so ladder snapshots are never
replaced by broad sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from rl.game_log_win import game_log_row_learner_win

FLAG_PRESENT = "_FLAG_PRESENT"

PROBE_OWNED_KEYS = frozenset({"--n-envs", "--n-steps", "--batch-size"})

# Human / hand-edited curriculum_state.json often uses "stage_d"; schedule uses long names.
_CURRICULUM_STAGE_SHORTHAND: dict[str, str] = {
    "stage_a": "stage_a_capture_bootstrap",
    "stage_b": "stage_b_capture_competent",
    "stage_c": "stage_c_terrain_competent",
    "stage_d": "stage_d_gl_std_map_pool_t3",
    "stage_e": "stage_e_gl_mixed_ladder",
    "stage_f": "stage_f_self_play_pure",
    "stage_g": "stage_g_mcts_eval_ready",
}


def normalize_curriculum_stage_name(name: str) -> str:
    """Map shorthand or legacy names to :data:`DEFAULT_SCHEDULE` stage ``name`` strings."""
    s = str(name).strip()
    if s == "stage_d_self_play_pure":
        return "stage_f_self_play_pure"
    return _CURRICULUM_STAGE_SHORTHAND.get(s, s)


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
class CurriculumState:
    current_stage_name: str
    games_observed_in_stage: int
    entered_stage_at_ts: float
    last_proposal_ts: float
    last_seen_finished_games: int = 0


@dataclass(slots=True)
class CurriculumProposal:
    stage_name: str
    args_overrides: dict[str, Any]
    reason: str
    metrics_snapshot: CompetenceMetrics


@dataclass(slots=True)
class CurriculumStage:
    name: str
    args_overrides: dict[str, Any]
    advance_when: Callable[[CompetenceMetrics], bool]
    min_games_in_stage: int
    description: str


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
    """Scalar 0..1 summarizing episode length + early-quit pressure (MCTS health–style)."""
    if avg_turns >= 25.0 and early_resign_rate <= 0.3:
        return 1.0
    len_part = max(0.0, min(1.0, avg_turns / 25.0))
    early_part = max(0.0, 1.0 - early_resign_rate)
    return max(0.0, min(1.0, 0.5 * len_part + 0.5 * early_part))


def compute_metrics(
    game_logs_path: Path, window_games: int = 200, machine_id: str | None = None
) -> CompetenceMetrics:
    rows = _finished_rows_for_machine(Path(game_logs_path), machine_id)
    if len(rows) > int(window_games):
        rows = rows[-int(window_games) :]
    n = len(rows)
    if n == 0:
        return CompetenceMetrics(
            games_in_window=0,
            capture_sense_score=0.0,
            terrain_usage_score=0.0,
            army_value_lead=0.0,
            win_rate=0.0,
            episode_quality=0.0,
            median_first_p0_capture_step=float("inf"),
            median_captures_completed_p0=0.0,
        )

    cap_scores: list[float] = []
    cap_counts: list[float] = []
    first_p0_steps: list[float] = []
    terr: list[float] = []
    army_pos: list[float] = []
    wins: list[float] = []
    turns: list[float] = []
    early: list[float] = []

    for r in rows:
        c = r.get("captures_completed_p0")
        try:
            c_f = float(c) if c is not None else 0.0
        except (TypeError, ValueError):
            c_f = 0.0
        cap_scores.append(min(1.0, c_f / 3.0))
        cap_counts.append(c_f)

        fcp = r.get("first_p0_capture_p0_step")
        if fcp is not None:
            try:
                first_p0_steps.append(float(fcp))
            except (TypeError, ValueError):
                pass

        tu = r.get("terrain_usage_p0")
        try:
            terr.append(float(tu) if tu is not None else 0.0)
        except (TypeError, ValueError):
            terr.append(0.0)

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
        try:
            t_f = float(t) if t is not None else 0.0
        except (TypeError, ValueError):
            t_f = 0.0
        turns.append(t_f)
        early.append(1.0 if t_f < 20.0 else 0.0)

    m_turns = _mean(turns)
    m_early = _mean(early)
    med_cap = _median_float(cap_counts)
    med_fc = _median_float(first_p0_steps)
    return CompetenceMetrics(
        games_in_window=n,
        capture_sense_score=_mean(cap_scores),
        terrain_usage_score=_mean(terr),
        army_value_lead=_mean(army_pos),
        win_rate=_mean(wins),
        episode_quality=_episode_quality_score(m_turns, m_early),
        median_first_p0_capture_step=(
            float("inf") if med_fc is None else float(med_fc)
        ),
        median_captures_completed_p0=float(med_cap) if med_cap is not None else 0.0,
    )


def _stage_index(schedule: list[CurriculumStage], name: str) -> int:
    for i, s in enumerate(schedule):
        if s.name == name:
            return i
    return 0


def classify_stage(
    metrics: CompetenceMetrics,
    current_state: CurriculumState,
    schedule: list[CurriculumStage],
) -> tuple[str, str]:
    """
    Return (stage_name, reason). Never decreases stage index below stage_a (index 0).
    Respects min_games_in_stage on the *current* stage before advancing by one step.
    """
    idx = _stage_index(
        schedule, normalize_curriculum_stage_name(current_state.current_stage_name)
    )
    idx = max(0, min(idx, len(schedule) - 1))
    stage = schedule[idx]

    if idx < len(schedule) - 1:
        if (
            current_state.games_observed_in_stage >= stage.min_games_in_stage
            and stage.advance_when(metrics)
        ):
            nxt = schedule[idx + 1]
            return nxt.name, f"advanced {stage.name} -> {nxt.name}: gates met"

    return stage.name, f"holding at {stage.name}: need more games or metrics"


DEFAULT_SCHEDULE: list[CurriculumStage] = [
    CurriculumStage(
        name="stage_a_capture_bootstrap",
        args_overrides={
            "--learner-greedy-mix": 0.3,
            "--cold-opponent": "greedy_capture",
            "--co-p0": 1,
            "--co-p1": 1,
            "--capture-move-gate": FLAG_PRESENT,
        },
        # §10g capture decay gate shape: fast first capture + sustained capture volume.
        advance_when=lambda m: (
            m.games_in_window >= 50
            and m.median_captures_completed_p0 >= 4.0
            and m.median_first_p0_capture_step <= 15.0
        ),
        min_games_in_stage=200,
        description="Cold start: greedy_capture opponent, capture-move gate, single Andy mirror",
    ),
    CurriculumStage(
        name="stage_b_capture_competent",
        args_overrides={
            "--learner-greedy-mix": 0.15,
            "--cold-opponent": "greedy_mix",
            "--co-p0": 1,
            "--co-p1": 1,
            "--capture-move-gate": FLAG_PRESENT,
        },
        advance_when=lambda m: (
            m.terrain_usage_score >= 0.52
            and m.capture_sense_score >= 0.5
            and m.win_rate >= 0.45
            and m.median_captures_completed_p0 >= 3.0
        ),
        min_games_in_stage=300,
        description="Capture solid; broaden cold opponent to greedy_mix",
    ),
    CurriculumStage(
        name="stage_c_terrain_competent",
        args_overrides={
            "--learner-greedy-mix": 0.05,
            "--cold-opponent": "greedy_mix",
            "--co-p0": 1,
            "--co-p1": 1,
            "--capture-move-gate": False,
        },
        advance_when=lambda m: (
            m.win_rate >= 0.62
            and m.terrain_usage_score >= 0.52
            and m.episode_quality >= 0.72
        ),
        min_games_in_stage=500,
        description="Terrain competent; drop capture-move gate; minimal greedy mix",
    ),
    CurriculumStage(
        name="stage_d_gl_std_map_pool_t3",
        args_overrides={
            "--map-id": None,
            "--tier": "T3",
            "--co-p0": 1,
            "--co-p1": 1,
            "--learner-greedy-mix": 0.05,
            "--cold-opponent": "greedy_mix",
            "--curriculum-broad-prob": 0.0,
            # Log label only; does not filter maps (sampling is uniform over GL Std in env).
            "--curriculum-tag": "gl_std_pool_t3_andy_mirror",
            # Explicit False: merge is ``merged_args.update(overrides)``; omitting the key
            # would leave ``--capture-move-gate`` stuck from stage_a/b in proposed_args.
            "--capture-move-gate": False,
        },
        advance_when=lambda m: (
            m.win_rate >= 0.58
            and m.terrain_usage_score >= 0.48
            and m.capture_sense_score >= 0.42
            and m.episode_quality >= 0.70
        ),
        min_games_in_stage=300,
        description=(
            "All GL Std maps (omit --map-id); fixed T3 Andy vs Andy per episode; "
            "not misery-only — old tag name *misery* was misleading. "
            "Use stage E+ for broad COs / 100% broad."
        ),
    ),
    CurriculumStage(
        name="stage_e_gl_mixed_ladder",
        args_overrides={
            "--map-id": None,
            "--tier": None,
            "--co-p0": None,
            "--co-p1": None,
            "--learner-greedy-mix": 0.0,
            "--cold-opponent": "random",
            "--curriculum-broad-prob": 1.0,
            "--curriculum-tag": "gl_mixed_ladder_full_random",
            "--capture-move-gate": False,
        },
        advance_when=lambda m: (
            m.win_rate >= 0.52
            and m.terrain_usage_score >= 0.42
            and m.episode_quality >= 0.68
        ),
        min_games_in_stage=400,
        description=(
            "Std maps; 100% broad (every episode full random map/tier/CO within pool). "
            "Live snapshot workers ignore broad."
        ),
    ),
    CurriculumStage(
        name="stage_f_self_play_pure",
        args_overrides={
            "--map-id": None,
            "--tier": None,
            "--co-p0": None,
            "--co-p1": None,
            "--learner-greedy-mix": 0.0,
            "--cold-opponent": "random",
            "--curriculum-broad-prob": 1.0,
            "--curriculum-tag": "gl_self_play_full_random",
            "--capture-move-gate": False,
        },
        advance_when=lambda m: (
            m.win_rate >= 0.52
            and m.terrain_usage_score >= 0.42
            and m.episode_quality >= 0.68
        ),
        min_games_in_stage=400,
        description=(
            "100% broad GL self-play mixture; advance to G when metrics match E→F bar "
            "(min 400 games in stage). Live snapshot workers ignore broad."
        ),
    ),
    CurriculumStage(
        name="stage_g_mcts_eval_ready",
        args_overrides={
            "--map-id": None,
            "--tier": None,
            "--co-p0": None,
            "--co-p1": None,
            "--learner-greedy-mix": 0.0,
            "--cold-opponent": "random",
            "--curriculum-broad-prob": 1.0,
            "--curriculum-tag": "gl_mixed_mcts_eval_full_random",
            "--capture-move-gate": False,
            # Storage + symmetric eval path; PPO rollouts still use the policy (train.py).
            "--mcts-mode": "eval_only",
            "--mcts-sims": 16,
        },
        advance_when=lambda m: False,
        min_games_in_stage=1_000_000,
        description=(
            "Terminal: 100% broad like F plus MCTS eval_only (baseline sims=16). "
            "Orchestrator mcts_health may overwrite --mcts-sims when gates pass. "
            "Live snapshot workers ignore broad."
        ),
    ),
]


def _strip_probe_owned(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in PROBE_OWNED_KEYS}


def _count_finished_games(path: Path, machine_id: str | None) -> int:
    return len(_finished_rows_for_machine(path, machine_id))


def _default_curriculum_state() -> CurriculumState:
    return CurriculumState(
        current_stage_name=DEFAULT_SCHEDULE[0].name,
        games_observed_in_stage=0,
        entered_stage_at_ts=0.0,
        last_proposal_ts=0.0,
        last_seen_finished_games=0,
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


def next_curriculum_state_after_tick(
    *,
    game_logs_path: Path,
    prev: CurriculumState,
    schedule: list[CurriculumStage] = DEFAULT_SCHEDULE,
    window_games: int = 200,
    machine_id: str | None = None,
    now_ts: float | None = None,
) -> CurriculumState:
    """Advance counters and possibly stage; used by orchestrator to persist curriculum_state.json."""
    import time as _time

    now = float(now_ts if now_ts is not None else _time.time())
    metrics = compute_metrics(game_logs_path, window_games=window_games, machine_id=machine_id)
    total_games = _count_finished_games(Path(game_logs_path), machine_id)
    delta = max(0, total_games - prev.last_seen_finished_games)

    new_stage, _reason = classify_stage(metrics, prev, schedule)
    if new_stage != prev.current_stage_name:
        return CurriculumState(
            current_stage_name=new_stage,
            games_observed_in_stage=delta,
            entered_stage_at_ts=now,
            last_proposal_ts=now,
            last_seen_finished_games=total_games,
        )
    return CurriculumState(
        current_stage_name=new_stage,
        games_observed_in_stage=prev.games_observed_in_stage + delta,
        entered_stage_at_ts=prev.entered_stage_at_ts,
        last_proposal_ts=now,
        last_seen_finished_games=total_games,
    )


def compute_proposal_stable(
    game_logs_path: Path,
    prev_state: CurriculumState,
    schedule: list[CurriculumStage] = DEFAULT_SCHEDULE,
    *,
    window_games: int = 200,
    machine_id: str | None = None,
    now_ts: float | None = None,
) -> tuple[CurriculumProposal, CurriculumState]:
    """
    Single call path: proposal args match ``next_curriculum_state_after_tick`` stage
    (classify uses pre-tick games_observed_in_stage).
    """
    metrics = compute_metrics(game_logs_path, window_games=window_games, machine_id=machine_id)
    new_stage, reason = classify_stage(metrics, prev_state, schedule)
    stage = schedule[_stage_index(schedule, new_stage)]
    overrides = _strip_probe_owned(dict(stage.args_overrides))
    for k in PROBE_OWNED_KEYS:
        assert k not in overrides, "curriculum must not override probe-owned keys"
    proposal = CurriculumProposal(
        stage_name=new_stage,
        args_overrides=overrides,
        reason=reason,
        metrics_snapshot=metrics,
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
    schedule: list[CurriculumStage] = DEFAULT_SCHEDULE,
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
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    gpath = args.game_log or (repo / "logs" / "game_log.jsonl")
    st: CurriculumState
    if args.state_file is not None:
        st = read_state(Path(args.state_file))
    else:
        st = _default_curriculum_state()
    prop, st_new = compute_proposal_stable(
        gpath,
        st,
        window_games=int(args.window_games),
        machine_id=args.machine_id,
    )
    out = {
        "proposal": asdict(prop),
        "metrics": asdict(prop.metrics_snapshot),
        "updated_state": asdict(st_new),
    }
    print(json.dumps(out, indent=2))
    if args.state_file is not None:
        write_state(Path(args.state_file), st_new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
