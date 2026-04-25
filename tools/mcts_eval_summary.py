# -*- coding: utf-8 -*-
"""Phase 11 Slice D — build :class:`tools.mcts_escalator.EscalatorCycleResult` from on-disk fleet eval verdicts.

Pure read helpers. The orchestrator (``scripts/fleet_orchestrator.py``)
calls :func:`build_cycle_result` once per pool machine after the MCTS
gate has flipped that machine to ``--mcts-mode != "off"``. The result
feeds :func:`tools.mcts_escalator.compute_sims_proposal`.

Inputs the helper looks at:

* ``<shared>/fleet/<machine_id>/eval/*.json`` — symmetric eval daemon
  verdicts (newest ``eval_window`` files by mtime; older files ignored).
  Aggregation uses :func:`rl.fleet_env.verdict_summary_from_symmetric_json`
  for tolerant parsing of ``candidate_wins`` / ``baseline_wins``.
* ``<shared>/logs/desync_register.jsonl`` — engine-vs-replay audit
  register written by ``tools/desync_audit.py``. As of schema_version 2
  (Phase 11d) each row carries ``machine_id`` and ``recorded_at`` so
  the escalator can attribute desyncs per-machine inside a real cycle
  window. Older rows (no per-row attribution) fall back to the file
  mtime as a coarse window proxy. See :func:`_count_recent_desyncs` for
  the exact filter rules.
* ``<shared>/fleet/<machine_id>/proposed_args.json`` — current
  ``--mcts-sims`` value (defaults to 16 if unset/missing).

Limitations / TODO:

* Train explained variance is scraped from SB3's TensorBoard event
  files via :func:`tools.tb_scrape_ev.latest_explained_variance`. When
  that returns ``None`` (no recent samples; cold boot or stale logs)
  we fall back to ``0.0`` so default thresholds keep DOUBLE blocked.
  The orchestrator (see ``scripts/fleet_orchestrator.py``) emits a
  separate ``mcts_ev_unavailable`` audit row in that case so callers
  can distinguish "EV missing" from "EV measured at 0.0".
* Per-machine desync attribution is strict only for schema_version 2
  rows; older registers still gate by file mtime (see above).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime as _datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mcts_baseline import MctsOffBaseline  # noqa: E402
from tools.mcts_escalator import EscalatorCycleResult  # noqa: E402
from tools.tb_scrape_ev import latest_explained_variance  # noqa: E402,F401

DEFAULT_EVAL_WINDOW: int = 200
DEFAULT_CYCLE_WINDOW_SECONDS: float = 3600.0
DEFAULT_FALLBACK_SIMS: int = 16
DEFAULT_EV_AGGREGATOR: str = "median"
DEFAULT_EV_WINDOW_SECONDS: float = 3600.0


def _load_verdict_summarizer() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Late import to avoid heavy ``rl.*`` work at module import time.

    Tests that monkeypatch ``rl.fleet_env`` after this module is loaded
    still see the latest function because we resolve it lazily.
    """
    from rl.fleet_env import verdict_summary_from_symmetric_json

    return verdict_summary_from_symmetric_json


def current_sims_from_proposed(machine_id: str, shared_root: Path) -> int:
    """Return the current ``--mcts-sims`` from ``proposed_args.json``.

    Falls back to :data:`DEFAULT_FALLBACK_SIMS` (16) for any error
    (missing file, malformed JSON, missing key, non-int value).
    """
    p = Path(shared_root) / "fleet" / str(machine_id) / "proposed_args.json"
    if not p.is_file():
        return DEFAULT_FALLBACK_SIMS
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_FALLBACK_SIMS
    args = raw.get("args") if isinstance(raw, dict) else None
    if not isinstance(args, dict):
        return DEFAULT_FALLBACK_SIMS
    val = args.get("--mcts-sims")
    if val is None:
        return DEFAULT_FALLBACK_SIMS
    try:
        return int(val)
    except (TypeError, ValueError):
        return DEFAULT_FALLBACK_SIMS


def _newest_eval_paths(
    machine_id: str, shared_root: Path, eval_window: int
) -> list[Path]:
    eval_dir = Path(shared_root) / "fleet" / str(machine_id) / "eval"
    if not eval_dir.is_dir():
        return []
    paths = [
        p for p in eval_dir.iterdir() if p.is_file() and p.suffix == ".json"
    ]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    n = max(0, int(eval_window))
    return paths[:n]


def _aggregate_verdicts(
    paths: list[Path],
    summarize: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[int, int, float]:
    """Return ``(verdicts_used, games_decided_total, winrate)``.

    ``winrate = candidate_wins / games_decided`` over all summed verdicts;
    ties (``baseline_wins`` only) reduce winrate as expected. Verdicts
    that fail to parse or to summarize are skipped silently.
    """
    cw_total = 0
    bw_total = 0
    used = 0
    for vp in paths:
        try:
            raw = json.loads(vp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        try:
            summ = summarize(raw)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(summ, dict):
            continue
        try:
            cw = int(summ.get("candidate_wins", 0))
            bw = int(summ.get("baseline_wins", 0))
        except (TypeError, ValueError):
            continue
        cw_total += cw
        bw_total += bw
        used += 1
    games = cw_total + bw_total
    wr = (cw_total / games) if games > 0 else 0.0
    return used, games, wr


def _count_recent_desyncs(
    shared_root: Path,
    machine_id: str,
    *,
    cycle_window_seconds: float,
    now_ts: float,
) -> int:
    """Count non-``ok`` rows in ``<shared>/logs/desync_register.jsonl``.

    Phase 11d (schema_version 2 of :class:`tools.desync_audit.AuditRow`)
    added per-row ``machine_id`` and ISO-8601 UTC ``recorded_at``. The
    counter prefers strict per-row filtering when those fields are
    present:

    * ``machine_id``: when a row carries the field, it must match
      ``machine_id`` to count. Rows missing the field are **dropped**
      from a per-machine count (legacy attribution would smear a
      neighbor's defect onto this host's DROP_TO_OFF gate).
    * ``recorded_at``: when a row carries the field, it must parse and
      lie within ``cycle_window_seconds`` of ``now_ts`` to count. Rows
      with an unparseable or missing timestamp fall through to the
      legacy file-mtime gate below.

    Legacy fallback: if **no** row in the file carries either field
    (pre-schema_version-2 register), gate the whole file by mtime — if
    the file was last modified within ``cycle_window_seconds`` we count
    every non-``ok`` row; otherwise return ``0``. This preserves the
    pre-Phase-11d behavior for older registers without smearing modern
    rows.
    """
    p = Path(shared_root) / "logs" / "desync_register.jsonl"
    if not p.is_file():
        return 0
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return 0
    rows: list[dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return 0
    if not rows:
        return 0

    schema_aware = any(
        ("machine_id" in r) or ("recorded_at" in r) for r in rows
    )
    cutoff = float(now_ts) - float(cycle_window_seconds)

    if not schema_aware:
        # Legacy fallback: pre-schema_version-2 register. Use file mtime
        # as the only window proxy and count every non-``ok`` row when
        # fresh. Stale files contribute zero.
        if now_ts - mtime > float(cycle_window_seconds):
            return 0
        n = 0
        for row in rows:
            cls = row.get("class")
            if cls is None:
                cls = row.get("cls")
            cls_s = str(cls or "").strip().lower()
            if cls_s and cls_s != "ok":
                n += 1
        return n

    n = 0
    target_mid = str(machine_id) if machine_id is not None else None
    for row in rows:
        cls = row.get("class")
        if cls is None:
            cls = row.get("cls")
        cls_s = str(cls or "").strip().lower()
        if not cls_s or cls_s == "ok":
            continue
        # machine_id filter: when target_mid is provided, drop rows that
        # either carry a different machine_id or carry no machine_id at
        # all (legacy rows cannot be attributed to this host).
        row_mid = row.get("machine_id")
        if target_mid is not None:
            if row_mid is None or str(row_mid) != target_mid:
                continue
        # recorded_at filter: when present and parseable, must be within
        # the cycle window. Unparseable / missing falls back to mtime.
        ts_str = row.get("recorded_at")
        ts_val: float | None = None
        if isinstance(ts_str, str) and ts_str:
            try:
                s = ts_str.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                ts_val = _datetime.fromisoformat(s).timestamp()
            except (ValueError, OSError):
                ts_val = None
        if ts_val is not None:
            if ts_val < cutoff:
                continue
        else:
            # Row without a parseable timestamp: gate by file mtime.
            if now_ts - mtime > float(cycle_window_seconds):
                continue
        n += 1
    return n


def build_cycle_result(
    machine_id: str,
    shared_root: Path,
    baseline: MctsOffBaseline | None,
    *,
    eval_window: int = DEFAULT_EVAL_WINDOW,
    cycle_window_seconds: float = DEFAULT_CYCLE_WINDOW_SECONDS,
    now_ts: float | None = None,
    ev_aggregator: str = DEFAULT_EV_AGGREGATOR,
    ev_window_seconds: float = DEFAULT_EV_WINDOW_SECONDS,
) -> EscalatorCycleResult | None:
    """Build the per-cycle metrics row consumed by the escalator.

    Returns ``None`` when:

    * ``baseline`` is ``None`` (the escalator must refuse to run without
      one — see ``tools/capture_mcts_baseline.py``).
    * The machine has no eval verdicts on disk yet, or every verdict
      failed to parse (and therefore games_decided would be zero).

    Otherwise returns a fully populated
    :class:`tools.mcts_escalator.EscalatorCycleResult`.

    ``explained_variance`` is scraped from SB3 TensorBoard event files
    via :func:`tools.tb_scrape_ev.latest_explained_variance` (resolved
    via ``sys.modules`` so test monkeypatches on this module take
    effect). When the scraper returns ``None`` we fall back to ``0.0``
    so default thresholds keep DOUBLE blocked. Distinguishing "EV
    missing" from "EV measured at 0.0" is the orchestrator's job — it
    calls the same scraper and emits ``mcts_ev_unavailable``.
    """
    if baseline is None:
        return None
    summarize = _load_verdict_summarizer()
    paths = _newest_eval_paths(machine_id, shared_root, eval_window)
    if not paths:
        return None
    used, games, wr = _aggregate_verdicts(paths, summarize)
    if used == 0 or games == 0:
        return None
    ts = float(now_ts) if now_ts is not None else time.time()
    desyncs = _count_recent_desyncs(
        Path(shared_root),
        str(machine_id),
        cycle_window_seconds=cycle_window_seconds,
        now_ts=ts,
    )
    sims = current_sims_from_proposed(str(machine_id), Path(shared_root))
    # Resolve via sys.modules so monkeypatches on this module's
    # ``latest_explained_variance`` attribute (test seam) are honored.
    _ev_fn = sys.modules[__name__].latest_explained_variance
    ev_value = _ev_fn(
        str(machine_id),
        Path(shared_root),
        recent_window_seconds=float(ev_window_seconds),
        aggregator=str(ev_aggregator),
        now_ts=now_ts,
    )
    train_explained_variance = float(ev_value) if ev_value is not None else 0.0
    return EscalatorCycleResult(
        cycle_ts=ts,
        sims=int(sims),
        winrate_vs_pool=float(wr),
        mcts_off_baseline=float(baseline.winrate_vs_pool),
        games_decided=int(games),
        explained_variance=float(train_explained_variance),
        engine_desyncs_in_cycle=int(desyncs),
        wall_s_per_decision_p50=0.0,
    )
