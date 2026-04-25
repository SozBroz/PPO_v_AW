# -*- coding: utf-8 -*-
"""Phase 10h: continuous fleet machine-health diagnosis classifier.

Reads game_log.jsonl filtered by machine_id, slots each machine into a health
state, and recommends a targeted intervention. Per plan §10h: ONE intervention
per machine per cycle, never auto-acts on the orchestrator host (pc-b) without
explicit operator opt-in, pathological always escalates to operator.

This session: classification + recommendation + audit-log only.
Orchestrator does NOT auto-apply interventions yet — that wires in next session
once the operator sees real diagnoses against their training run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from rl.game_log_win import game_log_row_learner_win

# ---------------------------------------------------------------------------
# Public structures
# ---------------------------------------------------------------------------


class DiagnosisState(str, Enum):
    """String values match plan §10h table (lowercase)."""

    FRESH = "fresh"
    BOOTSTRAPPING = "bootstrapping"
    COMPETENT = "competent"
    STUCK = "stuck"
    REGRESSING = "regressing"
    PATHOLOGICAL = "pathological"


@dataclass(slots=True)
class DiagnosisThresholds:
    recent_window: int = 200
    total_window: int = 500
    fresh_max_games: int = 100
    fresh_games_since_reload: int = 100
    competent_min_games: int = 200
    competent_winrate_min: float = 0.55
    regressing_win_drop: float = 0.15
    regressing_capture_median_drop: float = 2.0
    stuck_min_games: int = 500
    stuck_max_abs_winrate_delta: float = 0.03
    stuck_curriculum_flat_games: int = 500
    pathological_n_actions_ratio: float = 1.5
    pathological_winrate_flat_max: float = 0.05
    pathological_wall_ratio: float = 3.0
    # Absolute floor only when prior median is missing (insufficient history).
    pathological_wall_median_abs_s: float = 14_400.0
    pathological_opponent_dominance: float = 0.95
    intervention_cooldown_games: int = 200
    terminal_stage_name: str = "stage_f_self_play_pure"


DEFAULT_THRESHOLDS = DiagnosisThresholds()


@dataclass(slots=True)
class DiagnosisMetrics:
    games_in_window: int
    games_since_last_reload: int
    games_since_last_intervention: int
    winrate_recent_200: float
    winrate_prior_200: float
    winrate_delta_500: float
    captures_completed_p0_median_recent_200: float
    captures_completed_p0_median_prior_200: float
    n_actions_median_recent_200: float
    n_actions_median_prior_200: float
    episode_wall_s_median_recent_200: float
    episode_wall_s_median_prior_200: float
    opponent_type_distribution_recent_200: dict[str, float]
    opponent_type_distribution_prior_200: dict[str, float]
    curriculum_stage_name: str
    curriculum_games_observed_in_stage: int


@dataclass(slots=True)
class DiagnosisVerdict:
    machine_id: str
    measured_at_ts: str
    state: DiagnosisState
    metrics: DiagnosisMetrics
    recommended_intervention: str | None
    reason: str
    is_orchestrator_host: bool


# ---------------------------------------------------------------------------
# Log I/O (strict machine_id, same spirit as mcts_health)
# ---------------------------------------------------------------------------


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


def _finished_rows_for_machine(path: Path, machine_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _parse_log_lines(path):
        mid = row.get("machine_id")
        if mid is None or str(mid) != str(machine_id):
            continue
        if "turns" not in row:
            continue
        out.append(row)
    return out


def _reload_boundary_ts(
    applied_args_path: Path | None, curriculum_state_path: Path | None
) -> float | None:
    mtimes: list[float] = []
    for p in (applied_args_path, curriculum_state_path):
        if p is not None and p.is_file():
            try:
                mtimes.append(float(p.stat().st_mtime))
            except OSError:
                pass
    return max(mtimes) if mtimes else None


def _count_games_since_ts(rows: list[dict[str, Any]], since_ts: float | None) -> int:
    if since_ts is None:
        return len(rows)
    n = 0
    for r in rows:
        ts = r.get("timestamp")
        try:
            t = float(ts) if ts is not None else None
        except (TypeError, ValueError):
            t = None
        if t is None or t >= since_ts:
            n += 1
    return n


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.median(xs))


def _win_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    wins = sum(1 for r in rows if game_log_row_learner_win(r))
    return wins / len(rows)


def _float_field(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _captures_median(rows: list[dict[str, Any]]) -> float:
    xs = [_float_field(r, "captures_completed_p0") for r in rows]
    return _median(xs)


def _n_actions_median(rows: list[dict[str, Any]]) -> float:
    xs = [_float_field(r, "n_actions") for r in rows]
    return _median(xs)


def _wall_median(rows: list[dict[str, Any]]) -> float:
    xs = [_float_field(r, "episode_wall_s") for r in rows]
    return _median(xs)


def _opponent_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for r in rows:
        ot = r.get("opponent_type")
        k = str(ot) if ot is not None else "unknown"
        counts[k] = counts.get(k, 0) + 1
    n = len(rows)
    if n == 0:
        return {}
    return {k: v / n for k, v in sorted(counts.items())}


def _read_curriculum_sidecar(curriculum_state_path: Path | None) -> tuple[str, int]:
    if curriculum_state_path is None or not curriculum_state_path.is_file():
        return ("", 0)
    try:
        raw = json.loads(curriculum_state_path.read_text(encoding="utf-8"))
        name = str(raw.get("current_stage_name") or "")
        g = int(raw.get("games_observed_in_stage") or 0)
        return name, g
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return ("", 0)


def _last_intervention_game_count(history_path: Path | None) -> int | None:
    if history_path is None or not history_path.is_file():
        return None
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        return None
    last = entries[-1]
    if not isinstance(last, dict):
        return None
    try:
        return int(last.get("at_total_games"))
    except (TypeError, ValueError):
        return None


def compute_metrics(
    game_logs_path: Path | str,
    machine_id: str,
    recent_window: int = 200,
    total_window: int = 500,
    applied_args_path: Path | None = None,
    curriculum_state_path: Path | None = None,
    intervention_history_path: Path | None = None,
) -> DiagnosisMetrics:
    path = Path(game_logs_path)
    all_rows = _finished_rows_for_machine(path, machine_id)
    if len(all_rows) > total_window:
        window_rows = all_rows[-total_window:]
    else:
        window_rows = list(all_rows)

    n_win = len(window_rows)
    reload_ts = _reload_boundary_ts(applied_args_path, curriculum_state_path)
    games_since_reload = _count_games_since_ts(window_rows, reload_ts)

    total_all = len(all_rows)
    last_iv = _last_intervention_game_count(intervention_history_path)
    if last_iv is None:
        games_since_iv = total_all
    else:
        games_since_iv = max(0, total_all - last_iv)

    rw = min(recent_window, n_win)
    recent = window_rows[-rw:] if rw else []
    prior = window_rows[-2 * rw : -rw] if n_win >= 2 * rw else []

    wr_r = _win_rate(recent)
    wr_p = _win_rate(prior) if prior else wr_r
    cap_r = _captures_median(recent)
    cap_p = _captures_median(prior) if prior else cap_r
    na_r = _n_actions_median(recent)
    na_p = _n_actions_median(prior) if prior else na_r
    wall_r = _wall_median(recent)
    wall_p = _wall_median(prior) if prior else 0.0

    st_name, st_games = _read_curriculum_sidecar(curriculum_state_path)

    return DiagnosisMetrics(
        games_in_window=n_win,
        games_since_last_reload=games_since_reload,
        games_since_last_intervention=games_since_iv,
        winrate_recent_200=wr_r,
        winrate_prior_200=wr_p,
        winrate_delta_500=(wr_r - wr_p) if prior else 0.0,
        captures_completed_p0_median_recent_200=cap_r,
        captures_completed_p0_median_prior_200=cap_p,
        n_actions_median_recent_200=na_r,
        n_actions_median_prior_200=na_p,
        episode_wall_s_median_recent_200=wall_r,
        episode_wall_s_median_prior_200=wall_p,
        opponent_type_distribution_recent_200=_opponent_distribution(recent),
        opponent_type_distribution_prior_200=_opponent_distribution(prior),
        curriculum_stage_name=st_name,
        curriculum_games_observed_in_stage=st_games,
    )


def classify(
    metrics: DiagnosisMetrics, thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS
) -> tuple[DiagnosisState, str]:
    """Return (state, reason). Priority: PATHOLOGICAL → REGRESSING → STUCK → FRESH → BOOTSTRAPPING → COMPETENT."""

    tw = thresholds.total_window
    rw = thresholds.recent_window

    # --- PATHOLOGICAL ---
    path_reasons: list[str] = []
    prior = metrics.winrate_prior_200
    recent_wr = metrics.winrate_recent_200
    flat_wr = abs(metrics.winrate_recent_200 - metrics.winrate_prior_200) <= thresholds.pathological_winrate_flat_max

    na_p = metrics.n_actions_median_prior_200
    na_r = metrics.n_actions_median_recent_200
    if (
        metrics.games_in_window >= 2 * rw
        and na_p > 0
        and na_r >= thresholds.pathological_n_actions_ratio * na_p
        and flat_wr
    ):
        path_reasons.append(
            f"n_actions median {na_r:.1f} >= {thresholds.pathological_n_actions_ratio:.2f}x prior {na_p:.1f} with flat winrate"
        )

    wall_p = metrics.episode_wall_s_median_prior_200
    wall_r = metrics.episode_wall_s_median_recent_200
    if metrics.games_in_window >= 2 * rw and wall_p > 0.0:
        if wall_r >= thresholds.pathological_wall_ratio * wall_p:
            path_reasons.append(
                f"episode_wall_s median {wall_r:.1f}s >= {thresholds.pathological_wall_ratio:.1f}x prior {wall_p:.1f}s"
            )
    elif wall_p <= 0.0 and wall_r >= thresholds.pathological_wall_median_abs_s:
        path_reasons.append(
            f"episode_wall_s median recent {wall_r:.1f}s >= {thresholds.pathological_wall_median_abs_s:.0f}s (no prior window)"
        )

    # opponent distribution collapsed: one type dominates recent vs diversity in prior
    dist_r = metrics.opponent_type_distribution_recent_200
    dist_p = metrics.opponent_type_distribution_prior_200
    if dist_r and dist_p and metrics.games_in_window >= 2 * rw:
        max_r = max(dist_r.values())
        max_p = max(dist_p.values())
        if max_r >= thresholds.pathological_opponent_dominance and max_p < 0.8:
            path_reasons.append("opponent_type distribution collapsed (pool sampling likely stopped)")

    if path_reasons:
        return DiagnosisState.PATHOLOGICAL, "; ".join(path_reasons)

    # --- REGRESSING (needs enough games) ---
    if metrics.games_in_window >= tw:
        drop = metrics.winrate_prior_200 - metrics.winrate_recent_200
        cap_drop = (
            metrics.captures_completed_p0_median_prior_200
            - metrics.captures_completed_p0_median_recent_200
        )
        if drop >= thresholds.regressing_win_drop:
            return (
                DiagnosisState.REGRESSING,
                f"winrate dropped {drop:.2f} over {tw} games (prior {metrics.winrate_prior_200:.2f} → recent {metrics.winrate_recent_200:.2f})",
            )
        if cap_drop >= thresholds.regressing_capture_median_drop:
            return (
                DiagnosisState.REGRESSING,
                f"captures_completed_p0 median fell by {cap_drop:.2f} (prior {metrics.captures_completed_p0_median_prior_200:.2f} → recent {metrics.captures_completed_p0_median_recent_200:.2f})",
            )

    # --- STUCK ---
    if metrics.games_in_window >= thresholds.stuck_min_games:
        flat = abs(metrics.winrate_delta_500) <= thresholds.stuck_max_abs_winrate_delta
        curriculum_flat = (
            metrics.curriculum_games_observed_in_stage >= thresholds.stuck_curriculum_flat_games
            or (
                metrics.curriculum_stage_name == thresholds.terminal_stage_name
                and flat
            )
        )
        # No improvement over 500: flat winrate delta; curriculum not advancing — use games in stage as proxy
        if flat and curriculum_flat and metrics.curriculum_stage_name:
            return (
                DiagnosisState.STUCK,
                f"flat winrate (Δ={metrics.winrate_delta_500:+.3f}) over {tw} games and curriculum flat (stage={metrics.curriculum_stage_name}, games_in_stage={metrics.curriculum_games_observed_in_stage})",
            )
        if flat and not metrics.curriculum_stage_name and metrics.games_in_window >= tw:
            # no sidecar: conservative stuck only if very flat winrate over full window
            return (
                DiagnosisState.STUCK,
                f"flat winrate (Δ={metrics.winrate_delta_500:+.3f}) over {tw} games (no curriculum_state.json)",
            )

    # --- FRESH ---
    total_hint = metrics.games_in_window  # window-limited; cold start uses window size
    if total_hint < thresholds.fresh_max_games or metrics.games_since_last_reload < thresholds.fresh_games_since_reload:
        return (
            DiagnosisState.FRESH,
            f"cold start: games_in_window={total_hint}, games_since_last_reload={metrics.games_since_last_reload}",
        )

    # --- BOOTSTRAPPING (before COMPETENT: still on 10g schedule) ---
    term = thresholds.terminal_stage_name
    if metrics.curriculum_stage_name and metrics.curriculum_stage_name != term:
        return (
            DiagnosisState.BOOTSTRAPPING,
            f"curriculum stage {metrics.curriculum_stage_name} (not terminal {term})",
        )

    # --- COMPETENT ---
    if metrics.games_in_window >= thresholds.competent_min_games:
        if metrics.winrate_recent_200 >= thresholds.competent_winrate_min:
            return (
                DiagnosisState.COMPETENT,
                f"established ({metrics.games_in_window} games in window), winrate_recent={metrics.winrate_recent_200:.2f} >= {thresholds.competent_winrate_min}",
            )
        return (
            DiagnosisState.COMPETENT,
            f"established ({metrics.games_in_window} games in window); winrate_recent={metrics.winrate_recent_200:.2f} below ideal {thresholds.competent_winrate_min} but no regression/stuck signals (conservative)",
        )

    return (
        DiagnosisState.FRESH,
        f"insufficient window for competent ({metrics.games_in_window} < {thresholds.competent_min_games})",
    )


def _intervention_history_types(history_path: Path | None) -> list[str]:
    if history_path is None or not history_path.is_file():
        return []
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return []
    out: list[str] = []
    for e in entries:
        if isinstance(e, dict) and e.get("kind"):
            out.append(str(e["kind"]))
    return out


def recommend_intervention(
    state: DiagnosisState,
    metrics: DiagnosisMetrics,
    machine_id: str,
    intervention_history_path: Path | None = None,
    *,
    is_orchestrator_host: bool = False,
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
) -> tuple[str | None, str]:
    """Return (recommended_intervention, extra_reason_suffix for verdict.reason)."""
    if is_orchestrator_host:
        return None, "pc-b is orchestrator host; no auto-intervention without --apply-here"

    if state == DiagnosisState.PATHOLOGICAL:
        # Audit text lives in verdict.reason; no auto-intervention payload.
        return (
            None,
            "OPERATOR ALERT: pathological state — orchestrator frozen for this machine",
        )

    if state in (
        DiagnosisState.COMPETENT,
        DiagnosisState.BOOTSTRAPPING,
        DiagnosisState.FRESH,
    ):
        return None, ""

    cd = thresholds.intervention_cooldown_games
    if metrics.games_since_last_intervention < cd:
        return None, f"cooldown: {metrics.games_since_last_intervention} games since last intervention (need {cd})"

    hist = _intervention_history_types(intervention_history_path)

    if state == DiagnosisState.STUCK:
        if "stuck_ent_coef" not in hist:
            return (
                "Bump --ent-coef by +0.01 (cap 0.08) for exploration; restart train per 10f.",
                "",
            )
        if "stuck_greedy_mix" not in hist:
            return (
                "Re-enable --learner-greedy-mix 0.15 for ~200 games; then decay per 10g.",
                "",
            )
        if "stuck_curriculum_rollback" not in hist:
            return (
                "Rollback one curriculum step (broad_prob / tier); record rollback in fleet_curriculum_changes.jsonl.",
                "",
            )
        return (
            "Last resort: 10d hot reload weights from strongest pool member (cap weekly reloads per plan).",
            "",
        )

    if state == DiagnosisState.REGRESSING:
        n = sum(1 for k in hist if k.startswith("regress_"))
        if n == 0:
            return (
                "REGRESSING: rollback most recent curriculum change from logs/fleet_curriculum_changes.jsonl; wait 200 games.",
                "",
            )
        if n == 1:
            return (
                "REGRESSING: rollback the prior curriculum change; wait 200 games.",
                "",
            )
        return (
            "REGRESSING: two rollbacks insufficient — 10d weight reload from promoted latest.zip baseline.",
            "",
        )

    return None, ""


def compute_diagnosis(
    game_logs_path: Path | str,
    machine_id: str,
    *,
    is_orchestrator_host: bool = False,
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
    applied_args_path: Path | None = None,
    curriculum_state_path: Path | None = None,
    intervention_history_path: Path | None = None,
) -> DiagnosisVerdict:
    metrics = compute_metrics(
        game_logs_path,
        machine_id,
        recent_window=thresholds.recent_window,
        total_window=thresholds.total_window,
        applied_args_path=applied_args_path,
        curriculum_state_path=curriculum_state_path,
        intervention_history_path=intervention_history_path,
    )
    state, reason = classify(metrics, thresholds)
    rec, extra = recommend_intervention(
        state,
        metrics,
        machine_id,
        intervention_history_path,
        is_orchestrator_host=is_orchestrator_host,
        thresholds=thresholds,
    )
    if extra:
        reason = f"{reason}; {extra}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return DiagnosisVerdict(
        machine_id=machine_id,
        measured_at_ts=now,
        state=state,
        metrics=metrics,
        recommended_intervention=rec,
        reason=reason,
        is_orchestrator_host=is_orchestrator_host,
    )


def verdict_to_dict(v: DiagnosisVerdict) -> dict[str, Any]:
    d = asdict(v)
    d["state"] = v.state.value
    m = dict(d["metrics"])
    d["metrics"] = m
    d["schema_version"] = 1
    d["source"] = "tools/fleet_diagnosis.py"
    return d


def _metrics_from_dict(m: dict[str, Any]) -> DiagnosisMetrics:
    return DiagnosisMetrics(
        games_in_window=int(m["games_in_window"]),
        games_since_last_reload=int(m["games_since_last_reload"]),
        games_since_last_intervention=int(m["games_since_last_intervention"]),
        winrate_recent_200=float(m["winrate_recent_200"]),
        winrate_prior_200=float(m["winrate_prior_200"]),
        winrate_delta_500=float(m["winrate_delta_500"]),
        captures_completed_p0_median_recent_200=float(
            m["captures_completed_p0_median_recent_200"]
        ),
        captures_completed_p0_median_prior_200=float(
            m["captures_completed_p0_median_prior_200"]
        ),
        n_actions_median_recent_200=float(m["n_actions_median_recent_200"]),
        n_actions_median_prior_200=float(m["n_actions_median_prior_200"]),
        episode_wall_s_median_recent_200=float(m["episode_wall_s_median_recent_200"]),
        episode_wall_s_median_prior_200=float(
            m.get("episode_wall_s_median_prior_200", 0.0)
        ),
        opponent_type_distribution_recent_200=dict(
            m.get("opponent_type_distribution_recent_200") or {}
        ),
        opponent_type_distribution_prior_200=dict(
            m.get("opponent_type_distribution_prior_200") or {}
        ),
        curriculum_stage_name=str(m.get("curriculum_stage_name") or ""),
        curriculum_games_observed_in_stage=int(
            m.get("curriculum_games_observed_in_stage") or 0
        ),
    )


def parse_diagnosis_json(data: dict[str, Any]) -> DiagnosisVerdict | None:
    try:
        st_raw = str(data["state"])
        state = DiagnosisState(st_raw)
        m = _metrics_from_dict(dict(data["metrics"]))
        return DiagnosisVerdict(
            machine_id=str(data["machine_id"]),
            measured_at_ts=str(data["measured_at_ts"]),
            state=state,
            metrics=m,
            recommended_intervention=(
                None
                if data.get("recommended_intervention") is None
                else str(data["recommended_intervention"])
            ),
            reason=str(data.get("reason") or ""),
            is_orchestrator_host=bool(data.get("is_orchestrator_host", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def write_diagnosis(path: Path, verdict: DiagnosisVerdict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(verdict_to_dict(verdict), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix="diagnosis_", suffix=".json.tmp", dir=str(path.parent)
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


def read_diagnosis(path: Path) -> DiagnosisVerdict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return parse_diagnosis_json(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine-id", type=str, required=True)
    ap.add_argument(
        "--shared-root",
        type=Path,
        default=None,
        help="Fleet root (fleet/<id>/ sidecars). Defaults to repo root.",
    )
    ap.add_argument(
        "--game-log",
        type=Path,
        default=None,
        help="game_log.jsonl path (default: <repo>/logs/<machine-id>/game_log.jsonl or .../logs/game_log.jsonl)",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    ap.add_argument(
        "--orchestrator-host",
        action="store_true",
        help="Treat as pc-b (no actionable intervention text)",
    )
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Print JSON only; do not write diagnosis.json",
    )
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    shared = Path(args.shared_root).resolve() if args.shared_root else repo
    mid = str(args.machine_id)
    gpath = args.game_log
    if gpath is None:
        cand = repo / "logs" / mid / "game_log.jsonl"
        gpath = cand if cand.is_file() else repo / "logs" / "game_log.jsonl"
    fleet_d = shared / "fleet" / mid
    applied = fleet_d / "applied_args.json"
    curr = fleet_d / "curriculum_state.json"
    ivhist = fleet_d / "intervention_history.json"
    v = compute_diagnosis(
        gpath,
        mid,
        is_orchestrator_host=bool(args.orchestrator_host),
        applied_args_path=applied if applied.is_file() else None,
        curriculum_state_path=curr if curr.is_file() else None,
        intervention_history_path=ivhist if ivhist.is_file() else None,
    )
    d = verdict_to_dict(v)
    if args.print_only:
        print(json.dumps(d, indent=2, sort_keys=True))
        return 0
    dest = fleet_d / "diagnosis.json"
    write_diagnosis(dest, v)
    print(json.dumps(d, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
