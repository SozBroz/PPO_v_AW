# -*- coding: utf-8 -*-
"""Phase 11 Slice E — per-machine MCTS-off baseline I/O.

Pure helpers (no argparse, no subprocess) used by:

* ``tools/capture_mcts_baseline.py`` — CLI that runs the eval and writes the file.
* ``tools/mcts_escalator.py`` (Slice D, future) — reads the baseline so a cycle's
  ``EscalatorCycleResult.mcts_off_baseline`` reflects this machine's measured
  no-MCTS winrate vs the same opponent the eval daemon uses.

The on-disk artifact is ``<shared>/fleet/<machine_id>/mcts_off_baseline.json``
(written via :func:`write_baseline`, the canonical source of truth). Schema is
intentionally narrow: the escalator never has to inspect symmetric eval guts.

Atomic write contract mirrors :func:`tools.mcts_escalator.write_state`.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bump only on backwards-incompatible field changes; readers must reject older
# schemas explicitly rather than silently coerce.
MCTS_OFF_BASELINE_SCHEMA_VERSION: int = 1

DEFAULT_BASELINE_FILENAME = "mcts_off_baseline.json"
DEFAULT_MAX_AGE_HOURS: float = 168.0  # one week


@dataclass(slots=True)
class MctsOffBaseline:
    """Single-machine ``--mcts-mode off`` winrate snapshot.

    Field order matches the JSON shape declared in the Slice E composer prompt
    (and in the docstring of :func:`write_baseline`).
    """

    schema_version: int
    machine_id: str
    captured_at: str  # ISO-8601 UTC, e.g. ``"2026-04-23T12:30:00Z"``
    checkpoint_zip: str  # absolute path or shared-root relative
    checkpoint_zip_sha256: str
    games_decided: int
    winrate_vs_pool: float
    mcts_mode: str  # always ``"off"`` for Slice E baselines
    source: str  # producer name, e.g. ``"tools/capture_mcts_baseline.py"``


def utc_now_iso_z() -> str:
    """``datetime.utcnow``-style ISO with explicit Z suffix and no microseconds."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    # ``isoformat`` would emit ``+00:00``; the prompt's example uses ``Z``.
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def baseline_path(machine_id: str, shared_root: Path) -> Path:
    """Canonical on-disk location for *machine_id* under *shared_root*."""
    return Path(shared_root) / "fleet" / str(machine_id) / DEFAULT_BASELINE_FILENAME


def parse_baseline_json(data: dict[str, Any]) -> MctsOffBaseline | None:
    """Defensive parse. Returns ``None`` if any required field is missing or wrong-typed.

    Unknown extra keys are tolerated (forward compat). ``schema_version`` must
    equal :data:`MCTS_OFF_BASELINE_SCHEMA_VERSION` — older files force an
    operator re-capture rather than silent coerce.
    """
    if not isinstance(data, dict):
        return None
    try:
        sv = int(data["schema_version"])
        if sv != MCTS_OFF_BASELINE_SCHEMA_VERSION:
            return None
        mid = str(data["machine_id"])
        captured_at = str(data["captured_at"])
        checkpoint_zip = str(data["checkpoint_zip"])
        sha = str(data["checkpoint_zip_sha256"])
        games = int(data["games_decided"])
        wr = float(data["winrate_vs_pool"])
        mode = str(data["mcts_mode"])
        source = str(data["source"])
    except (KeyError, TypeError, ValueError):
        return None
    if not mid or not captured_at or not checkpoint_zip:
        return None
    if games < 0:
        return None
    if not (0.0 <= wr <= 1.0):
        return None
    return MctsOffBaseline(
        schema_version=sv,
        machine_id=mid,
        captured_at=captured_at,
        checkpoint_zip=checkpoint_zip,
        checkpoint_zip_sha256=sha,
        games_decided=games,
        winrate_vs_pool=wr,
        mcts_mode=mode,
        source=source,
    )


def read_baseline(machine_id: str, shared_root: Path) -> MctsOffBaseline | None:
    """Read ``<shared>/fleet/<machine_id>/mcts_off_baseline.json``.

    Returns ``None`` if the file is missing, unreadable, malformed JSON, or
    fails :func:`parse_baseline_json` validation. Callers should treat ``None``
    as "no baseline; re-capture before escalating".
    """
    p = baseline_path(machine_id, shared_root)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parse_baseline_json(raw)


def _parse_iso_z(value: str) -> datetime | None:
    """Parse ``"YYYY-MM-DDTHH:MM:SSZ"`` (and ``+00:00`` variants) to a UTC ``datetime``."""
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_baseline_stale(
    baseline: MctsOffBaseline,
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> bool:
    """True iff ``captured_at`` is **strictly older** than *max_age_hours*.

    Boundary contract: when ``age_hours == max_age_hours`` the baseline is
    NOT stale. This keeps "exactly one week" from flapping to stale just
    because the orchestrator polled half a second late.

    Unparseable ``captured_at`` is treated as stale (force re-capture).
    """
    dt = _parse_iso_z(baseline.captured_at)
    if dt is None:
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_hours = (now - dt).total_seconds() / 3600.0
    return age_hours > float(max_age_hours)


def write_baseline(baseline: MctsOffBaseline, fleet_dir: Path) -> Path:
    """Atomically write *baseline* into ``<fleet_dir>/mcts_off_baseline.json``.

    ``fleet_dir`` is the per-machine fleet directory
    (``<shared>/fleet/<machine_id>``); the function ``mkdir -p``s it so callers
    do not need to bootstrap. Returns the final path.

    Atomicity is via tempfile + :func:`os.replace`, same pattern as
    :func:`tools.mcts_escalator.write_state` — no half-written readers ever.
    """
    fleet_dir = Path(fleet_dir)
    fleet_dir.mkdir(parents=True, exist_ok=True)
    out = fleet_dir / DEFAULT_BASELINE_FILENAME
    payload = json.dumps(asdict(baseline), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix="mcts_off_baseline_", suffix=".json.tmp", dir=str(fleet_dir)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_name, out)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return out
