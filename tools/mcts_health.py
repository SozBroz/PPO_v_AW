# -*- coding: utf-8 -*-
"""
MCTS health gate: rolling-window competence metrics from ``game_log.jsonl``.

Computes a conservative per-machine recommendation for whether MCTS ``eval_only``
is appropriate. Does **not** change training defaults; operators apply manually.

Environment variables (override defaults; all optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Window and minimum sample size**

- ``AWBW_MCTS_HEALTH_WINDOW`` — rolling game count (default ``200``).
- ``AWBW_MCTS_HEALTH_MIN_GAMES`` — require at least this many games or verdict
  is "insufficient data" (default ``50``).

**Thresholds (pass if metric meets threshold; see table in module / Phase 11d plan)**

- ``AWBW_MCTS_HEALTH_CAPTURE_SENSE_MIN`` (default ``0.4``)
- ``AWBW_MCTS_HEALTH_CAPTURE_COMPLETIONS_MIN`` (default ``1.5``) —
  :attr:`MctsHealthMetrics.avg_capture_completions_per_game`
- ``AWBW_MCTS_HEALTH_TERRAIN_MIN`` (default ``0.5``)
- ``AWBW_MCTS_HEALTH_ARMY_VALUE_LEAD_MIN`` (default ``0.45``)
- ``AWBW_MCTS_HEALTH_WIN_RATE_MIN`` (default ``0.4``) — for ``pass_army_value``
- ``AWBW_MCTS_HEALTH_EPISODE_LEN_MIN`` (default ``25.0``) — mean turns
- ``AWBW_MCTS_HEALTH_EARLY_RESIGN_MAX`` (default ``0.3``) — max fraction of
  games with ``turns < 20``

**MCTS sims escalator** (higher wins / captures / army lead → more sims)

- ``AWBW_MCTS_HEALTH_SIMS_TIER1_WIN`` (default ``0.4``) — minimum win rate for
  sims=8 (with ``pass_overall``).
- ``AWBW_MCTS_HEALTH_SIMS_TIER2_WIN`` (default ``0.5``)
- ``AWBW_MCTS_HEALTH_SIMS_TIER2_CAPTURE`` (default ``2.5``) — for sims=16
- ``AWBW_MCTS_HEALTH_SIMS_TIER3_WIN`` (default ``0.55``)
- ``AWBW_MCTS_HEALTH_SIMS_TIER3_ARMY_LEAD`` (default ``0.55``) — for sims=32

Metric sources and approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Win rate, episode length, early "resign"** — from ``winner``, ``turns``:
  fully defined on schema ≥1.7 finished-game rows (1.8 adds ``truncated`` / ``truncation_reason``).

- **Capture** — true "fraction of P0 turns with contested capturable property
  while eligible infantry exists" is **not** in the log. We approximate with
  ``mean(min(1, captures_completed_p0/3))`` per game, using
  ``captures_completed_p0`` from the finished-game record. *TODO: per-turn
  contested-capture flag in ``game_log.jsonl`` for the exact metric.*

- **Terrain** — the plan asked for "fraction of P0 unit-turns on >=1* terrain
  when better was in move range." The schema provides ``terrain_usage_p0``:
  fraction of *live* P0 units on defense ``>=2`` *at episode end* only. We use
  the mean of that field as ``avg_terrain_usage_score``, which correlates with
  defensive placement but is **not** per-turn. *TODO: optional per-episode
  roll-up in schema if the end snapshot is too weak.*

- **Army value lead** — no ``army_value_delta`` field. We use HP exchange:
  ``losses_hp[1] > losses_hp[0]`` (P1 lost more HP than P0) as a positive
  engagement proxy. *TODO: end-game army value or cumulative value lead if
  added to schema.*
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MctsHealthMetrics:
    games_in_window: int
    capture_sense_score: float
    avg_capture_completions_per_game: float
    avg_terrain_usage_score: float
    army_value_lead_pos_rate: float
    win_rate: float
    avg_episode_length_turns: float
    early_resign_rate: float


@dataclass(slots=True)
class MctsHealthVerdict:
    machine_id: str
    measured_at: str
    games_in_window: int
    metrics: MctsHealthMetrics
    pass_capture: bool
    pass_terrain: bool
    pass_army_value: bool
    pass_episode_quality: bool
    pass_overall: bool
    proposed_mcts_mode: str
    proposed_mcts_sims: int
    reasoning: str


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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


def _rows_for_machine(
    path: Path, machine_id: str
) -> list[dict[str, Any]]:
    """
    Load finished-game rows for *machine_id*.

    Rows without ``machine_id`` or with a mismatched value are **skipped** so
    mixed or legacy files do not pollute a machine's window (conservative).
    """
    out: list[dict[str, Any]] = []
    for row in _parse_log_lines(path):
        mid = row.get("machine_id")
        if mid is not None and str(mid) != str(machine_id):
            continue
        if mid is None:
            # Strict: do not count un-stamped rows toward this machine
            continue
        # Finished-game records have ``winner`` and ``turns``
        # Finished-game writer always sets ``turns``; ignore stray lines
        if "turns" not in row:
            continue
        out.append(row)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _compute_sims_tier(
    *,
    pass_overall: bool,
    win_rate: float,
    avg_captures: float,
    army_value_lead_pos_rate: float,
) -> int:
    if not pass_overall:
        return 0
    t1 = _env_float("AWBW_MCTS_HEALTH_SIMS_TIER1_WIN", 0.4)
    t2w = _env_float("AWBW_MCTS_HEALTH_SIMS_TIER2_WIN", 0.5)
    t2c = _env_float("AWBW_MCTS_HEALTH_SIMS_TIER2_CAPTURE", 2.5)
    t3w = _env_float("AWBW_MCTS_HEALTH_SIMS_TIER3_WIN", 0.55)
    t3a = _env_float("AWBW_MCTS_HEALTH_SIMS_TIER3_ARMY_LEAD", 0.55)
    if win_rate < t1:
        return 0
    if (
        win_rate >= t3w
        and army_value_lead_pos_rate >= t3a
    ):
        return 32
    if win_rate >= t2w and avg_captures >= t2c:
        return 16
    return 8


def compute_health(machine_id: str, logs_dir: Path, window: int = 200) -> MctsHealthVerdict:
    window = int(_env_int("AWBW_MCTS_HEALTH_WINDOW", window))
    min_games = int(_env_int("AWBW_MCTS_HEALTH_MIN_GAMES", 50))

    cap_sense_min = _env_float("AWBW_MCTS_HEALTH_CAPTURE_SENSE_MIN", 0.4)
    cap_comp_min = _env_float("AWBW_MCTS_HEALTH_CAPTURE_COMPLETIONS_MIN", 1.5)
    terrain_min = _env_float("AWBW_MCTS_HEALTH_TERRAIN_MIN", 0.5)
    army_lead_min = _env_float("AWBW_MCTS_HEALTH_ARMY_VALUE_LEAD_MIN", 0.45)
    win_min = _env_float("AWBW_MCTS_HEALTH_WIN_RATE_MIN", 0.4)
    ep_len_min = _env_float("AWBW_MCTS_HEALTH_EPISODE_LEN_MIN", 25.0)
    early_resign_max = _env_float("AWBW_MCTS_HEALTH_EARLY_RESIGN_MAX", 0.3)

    path = Path(logs_dir) / "game_log.jsonl"
    all_rows = _rows_for_machine(path, machine_id)
    if len(all_rows) > window:
        all_rows = all_rows[-window:]

    n = len(all_rows)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if n < min_games:
        metrics = MctsHealthMetrics(
            games_in_window=n,
            capture_sense_score=0.0,
            avg_capture_completions_per_game=0.0,
            avg_terrain_usage_score=0.0,
            army_value_lead_pos_rate=0.0,
            win_rate=0.0,
            avg_episode_length_turns=0.0,
            early_resign_rate=0.0,
        )
        return MctsHealthVerdict(
            machine_id=machine_id,
            measured_at=now_iso,
            games_in_window=n,
            metrics=metrics,
            pass_capture=False,
            pass_terrain=False,
            pass_army_value=False,
            pass_episode_quality=False,
            pass_overall=False,
            proposed_mcts_mode="off",
            proposed_mcts_sims=0,
            reasoning="insufficient data",
        )

    cap_scores: list[float] = []
    caps: list[float] = []
    terr: list[float] = []
    army_pos: list[float] = []
    wins: list[float] = []
    turns: list[float] = []
    early: list[float] = []

    for r in all_rows:
        c = r.get("captures_completed_p0")
        try:
            c_f = float(c) if c is not None else 0.0
        except (TypeError, ValueError):
            c_f = 0.0
        caps.append(c_f)
        # Proxy: normalize per-game capture pressure (see module docstring)
        cap_scores.append(min(1.0, c_f / 3.0))

        tu = r.get("terrain_usage_p0")
        try:
            terr.append(float(tu) if tu is not None else 0.0)
        except (TypeError, ValueError):
            terr.append(0.0)

        lh = r.get("losses_hp")
        pos = 0.0
        if (
            isinstance(lh, (list, tuple))
            and len(lh) >= 2
        ):
            try:
                p0l, p1l = float(lh[0]), float(lh[1])
                pos = 1.0 if p1l > p0l else 0.0
            except (TypeError, ValueError):
                pos = 0.0
        army_pos.append(pos)

        w = r.get("winner")
        try:
            wi = int(w) if w is not None else -99
        except (TypeError, ValueError):
            wi = -99
        wins.append(1.0 if wi == 0 else 0.0)

        t = r.get("turns")
        try:
            t_f = float(t) if t is not None else 0.0
        except (TypeError, ValueError):
            t_f = 0.0
        turns.append(t_f)
        early.append(1.0 if t_f < 20.0 else 0.0)

    m_capture_sense = _mean(cap_scores)
    m_caps = _mean(caps)
    m_terrain = _mean(terr)
    m_army = _mean(army_pos)
    m_wr = _mean(wins)
    m_turns = _mean(turns)
    m_early = _mean(early)

    metrics = MctsHealthMetrics(
        games_in_window=n,
        capture_sense_score=m_capture_sense,
        avg_capture_completions_per_game=m_caps,
        avg_terrain_usage_score=m_terrain,
        army_value_lead_pos_rate=m_army,
        win_rate=m_wr,
        avg_episode_length_turns=m_turns,
        early_resign_rate=m_early,
    )

    pass_capture = m_capture_sense >= cap_sense_min and m_caps >= cap_comp_min
    pass_terrain = m_terrain >= terrain_min
    pass_army_value = m_army >= army_lead_min and m_wr >= win_min
    pass_epis = m_turns >= ep_len_min and m_early <= early_resign_max
    pass_all = (
        pass_capture
        and pass_terrain
        and pass_army_value
        and pass_epis
    )

    caveats: list[str] = [
        "capture_sense: proxy from captures_completed_p0 (not per-turn contested), "
        "see tools/mcts_health.py docstring",
        "terrain: mean terrain_usage_p0 (end-of-episode snapshot, defense>=2), not per-turn",
        "army value: proxy via losses_hp trade (P1 lost HP > P0 lost HP), not unit value",
    ]
    reason_parts: list[str] = []
    if pass_all:
        reason_parts.append("all four threshold groups passed")
    else:
        if not pass_capture:
            reason_parts.append("failed capture")
        if not pass_terrain:
            reason_parts.append("failed terrain")
        if not pass_army_value:
            reason_parts.append("failed army value / win rate")
        if not pass_epis:
            reason_parts.append("failed episode quality")
    reason_parts.append(" | " + " ".join(caveats))
    reasoning = "; ".join(reason_parts)

    if pass_all:
        mode = "eval_only"
    else:
        mode = "off"

    sims = _compute_sims_tier(
        pass_overall=pass_all,
        win_rate=m_wr,
        avg_captures=m_caps,
        army_value_lead_pos_rate=m_army,
    )

    return MctsHealthVerdict(
        machine_id=machine_id,
        measured_at=now_iso,
        games_in_window=n,
        metrics=metrics,
        pass_capture=pass_capture,
        pass_terrain=pass_terrain,
        pass_army_value=pass_army_value,
        pass_episode_quality=pass_epis,
        pass_overall=pass_all,
        proposed_mcts_mode=mode,
        proposed_mcts_sims=sims,
        reasoning=reasoning,
    )


def _parse_iso_utc(s: str) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def is_mcts_health_stale(
    measured_at: str, *, now: datetime | None = None, max_age_hours: float = 24.0
) -> bool:
    """
    A verdict older than *max_age_hours* must not drive orchestrator
    decisions (treat as ``mode=off``).
    """
    dt = _parse_iso_utc(measured_at)
    if dt is None:
        return True
    tnow = now or datetime.now(timezone.utc)
    if tnow.tzinfo is None:
        tnow = tnow.replace(tzinfo=timezone.utc)
    age_s = (tnow - dt).total_seconds()
    return age_s > max_age_hours * 3600.0


def parse_mcts_health_json(data: dict[str, Any]) -> MctsHealthVerdict | None:
    """Reconstruct a verdict written by :func:`verdict_to_dict` / ``mcts_health.json``."""
    try:
        m = data["metrics"]
        metrics = MctsHealthMetrics(
            games_in_window=int(m["games_in_window"]),
            capture_sense_score=float(m["capture_sense_score"]),
            avg_capture_completions_per_game=float(
                m["avg_capture_completions_per_game"]
            ),
            avg_terrain_usage_score=float(m["avg_terrain_usage_score"]),
            army_value_lead_pos_rate=float(m["army_value_lead_pos_rate"]),
            win_rate=float(m["win_rate"]),
            avg_episode_length_turns=float(m["avg_episode_length_turns"]),
            early_resign_rate=float(m["early_resign_rate"]),
        )
        return MctsHealthVerdict(
            machine_id=str(data["machine_id"]),
            measured_at=str(data["measured_at"]),
            games_in_window=int(data["games_in_window"]),
            metrics=metrics,
            pass_capture=bool(data["pass_capture"]),
            pass_terrain=bool(data["pass_terrain"]),
            pass_army_value=bool(data["pass_army_value"]),
            pass_episode_quality=bool(data["pass_episode_quality"]),
            pass_overall=bool(data["pass_overall"]),
            proposed_mcts_mode=str(data["proposed_mcts_mode"]),
            proposed_mcts_sims=int(data["proposed_mcts_sims"]),
            reasoning=str(data.get("reasoning") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def stale_mcts_off_verdict(base: MctsHealthVerdict) -> MctsHealthVerdict:
    """Operator-facing override: stale file → MCTS off regardless of prior metrics."""
    return MctsHealthVerdict(
        machine_id=base.machine_id,
        measured_at=base.measured_at,
        games_in_window=base.games_in_window,
        metrics=base.metrics,
        pass_capture=False,
        pass_terrain=False,
        pass_army_value=False,
        pass_episode_quality=False,
        pass_overall=False,
        proposed_mcts_mode="off",
        proposed_mcts_sims=0,
        reasoning="stale verdict",
    )


def verdict_to_dict(v: MctsHealthVerdict) -> dict[str, Any]:
    d = asdict(v)
    d["schema_version"] = 1
    d["source"] = "tools/mcts_health.py"
    return d


def write_health_json(verdict: MctsHealthVerdict, fleet_dir: Path) -> Path:
    """
    Atomically write ``mcts_health.json`` under *fleet_dir* (the machine
    directory, e.g. ``.../fleet/pc-b``).
    """
    fleet_dir = Path(fleet_dir)
    fleet_dir.mkdir(parents=True, exist_ok=True)
    dest = fleet_dir / "mcts_health.json"
    payload = json.dumps(verdict_to_dict(verdict), indent=2)
    # Atomic replace on the same volume
    fd, tmp_name = tempfile.mkstemp(
        prefix="mcts_health_", suffix=".json.tmp", dir=str(fleet_dir)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        tmp_path = Path(tmp_name)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return dest


def _default_logs_dir(repo_root: Path, machine_id: str) -> Path:
    return repo_root / "logs" / machine_id


def _default_fleet_dir(repo_root: Path, machine_id: str) -> Path:
    return repo_root / "fleet" / machine_id


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine-id", type=str, required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory containing game_log.jsonl (default: <repo>/logs/<machine_id>)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, directory to write mcts_health.json (default: <repo>/fleet/<machine_id>)",
    )
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Print JSON to stdout; do not write a file",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (for default paths)",
    )
    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve()
    logs_dir = args.logs_dir
    if logs_dir is None:
        logs_dir = _default_logs_dir(repo_root, args.machine_id)
    v = compute_health(str(args.machine_id), Path(logs_dir), window=int(args.window))
    d = verdict_to_dict(v)
    if args.print_only:
        print(json.dumps(d, indent=2))
        return 0
    if args.out is not None:
        fleet_dir = Path(args.out)
    else:
        fleet_dir = _default_fleet_dir(repo_root, args.machine_id)
    path = write_health_json(v, fleet_dir)
    print(str(path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
