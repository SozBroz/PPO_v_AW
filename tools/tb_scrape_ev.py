# -*- coding: utf-8 -*-
"""Phase 11d EV scrape — read ``train/explained_variance`` from SB3 TensorBoard event files.

Stable-Baselines3's :class:`MaskablePPO.train()` publishes the scalar
``train/explained_variance`` to TensorBoard each rollout. This helper
locates the most recent event files under ``<shared>/logs[/<machine_id>]``,
reads that scalar, and aggregates samples within a recent wall-time
window so the MCTS escalator (see :mod:`tools.mcts_escalator`) can gate
sim-budget DOUBLEs on policy-value learning quality instead of being
permanently pinned at HOLD.

The canonical reader is
``tensorboard.backend.event_processing.event_accumulator.EventAccumulator``
(``tensorboard`` is a hard dep of this repo — see ``requirements.txt``).
If for some reason the package is unavailable at import time, we fall
back to ``tensorflow.compat.v1.train.summary_iterator``; if neither is
available we return ``None`` and the escalator gate stays HOLD.

Module entry points used by callers:

* :func:`find_tb_event_files`     — recursive ``events.out.tfevents.*`` discovery
* :func:`scrape_explained_variance` — pull samples for a tag across files
* :func:`aggregate_recent_ev`     — median / mean / last over a sample list
* :func:`latest_explained_variance` — convenience entry point used by the
  orchestrator-side escalator wiring (returns ``None`` on no signal).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCALAR_TAG: str = "train/explained_variance"
DEFAULT_WINDOW_SECONDS: float = 3600.0
DEFAULT_MAX_AGE_HOURS: float = 24.0
DEFAULT_AGGREGATOR: str = "median"
_VALID_AGGREGATORS = ("median", "mean", "last")


@dataclass(slots=True)
class ExplainedVarianceSample:
    """One scalar sample read from an event file."""

    wall_time: float
    step: int
    value: float


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_tb_event_files(
    logs_dir: Path,
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    now_ts: float | None = None,
) -> list[Path]:
    """Find ``events.out.tfevents.*`` recursively under ``logs_dir``.

    Filters by file mtime within ``max_age_hours``. Returned newest-first.
    Returns ``[]`` if ``logs_dir`` is missing or contains no matching files.
    """
    root = Path(logs_dir)
    if not root.is_dir():
        return []
    cutoff = (
        float(now_ts) if now_ts is not None else time.time()
    ) - float(max_age_hours) * 3600.0
    out: list[tuple[float, Path]] = []
    for p in root.rglob("events.out.tfevents.*"):
        try:
            st = p.stat()
        except OSError:
            continue
        if not p.is_file():
            continue
        if st.st_mtime < cutoff:
            continue
        out.append((st.st_mtime, p))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return [p for _mt, p in out]


# ---------------------------------------------------------------------------
# Scalar reader
# ---------------------------------------------------------------------------


def _read_one_eventfile_with_accumulator(
    path: Path, scalar_tag: str
) -> list[ExplainedVarianceSample]:
    """Read scalar values from a single event file via ``EventAccumulator``.

    Returns ``[]`` for any failure (missing tag, corrupt/empty file).
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator  # type: ignore
    except Exception:  # noqa: BLE001
        return _read_one_eventfile_with_summary_iterator(path, scalar_tag)
    try:
        ea = event_accumulator.EventAccumulator(
            str(path),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
    except Exception:  # noqa: BLE001
        return []
    try:
        tags = ea.Tags().get("scalars", []) or []
    except Exception:  # noqa: BLE001
        return []
    if scalar_tag not in tags:
        return []
    try:
        events = ea.Scalars(scalar_tag)
    except Exception:  # noqa: BLE001
        return []
    out: list[ExplainedVarianceSample] = []
    for ev in events:
        try:
            out.append(
                ExplainedVarianceSample(
                    wall_time=float(ev.wall_time),
                    step=int(ev.step),
                    value=float(ev.value),
                )
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def _read_one_eventfile_with_summary_iterator(
    path: Path, scalar_tag: str
) -> list[ExplainedVarianceSample]:
    """Fallback reader using ``tensorflow.compat.v1.train.summary_iterator``.

    Returns ``[]`` if TF is not installed or the file is corrupt.
    """
    try:
        from tensorflow.compat.v1.train import summary_iterator  # type: ignore
    except Exception:  # noqa: BLE001
        return []
    out: list[ExplainedVarianceSample] = []
    try:
        for ev in summary_iterator(str(path)):
            try:
                summary = ev.summary
                wall = float(getattr(ev, "wall_time", 0.0))
                step = int(getattr(ev, "step", 0))
            except Exception:  # noqa: BLE001
                continue
            for value in getattr(summary, "value", []):
                try:
                    if value.tag != scalar_tag:
                        continue
                    out.append(
                        ExplainedVarianceSample(
                            wall_time=wall,
                            step=step,
                            value=float(value.simple_value),
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    continue
    except Exception:  # noqa: BLE001
        return out
    return out


def scrape_explained_variance(
    event_files: Iterable[Path],
    *,
    scalar_tag: str = DEFAULT_SCALAR_TAG,
    recent_window_seconds: float = DEFAULT_WINDOW_SECONDS,
    now_ts: float | None = None,
) -> list[ExplainedVarianceSample]:
    """Read ``scalar_tag`` from each event file; keep samples within window.

    A sample is "recent" when ``now - sample.wall_time <= recent_window_seconds``.
    Robust to corrupt/empty files (skipped silently — no exception escapes).
    Returned newest-first by wall_time.
    """
    cutoff = (
        float(now_ts) if now_ts is not None else time.time()
    ) - float(recent_window_seconds)
    samples: list[ExplainedVarianceSample] = []
    for f in event_files:
        try:
            chunk = _read_one_eventfile_with_accumulator(Path(f), scalar_tag)
        except Exception:  # noqa: BLE001
            continue
        for s in chunk:
            if s.wall_time >= cutoff:
                samples.append(s)
    samples.sort(key=lambda s: s.wall_time, reverse=True)
    return samples


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_recent_ev(
    samples: list[ExplainedVarianceSample],
    *,
    aggregator: str = DEFAULT_AGGREGATOR,
) -> float | None:
    """Aggregate ``samples`` into a single float; ``None`` on empty input.

    ``aggregator``: one of ``median``, ``mean``, ``last`` (newest by
    wall_time). Unknown aggregator raises :class:`ValueError`.
    """
    if not samples:
        return None
    agg = str(aggregator).strip().lower()
    if agg not in _VALID_AGGREGATORS:
        raise ValueError(
            f"unknown aggregator {aggregator!r}; expected one of {_VALID_AGGREGATORS}"
        )
    values = [float(s.value) for s in samples]
    if agg == "median":
        return float(statistics.median(values))
    if agg == "mean":
        return float(statistics.fmean(values))
    # last == newest by wall_time; samples already sorted newest-first
    newest = max(samples, key=lambda s: s.wall_time)
    return float(newest.value)


# ---------------------------------------------------------------------------
# Convenience entry point used by the escalator wiring
# ---------------------------------------------------------------------------


def _candidate_logs_dirs(machine_id: str, shared_root: Path) -> list[Path]:
    """Lookup order for SB3 TB events.

    1. ``<shared>/logs/<machine_id>`` — multi-machine layout (future-facing;
       Composer's spec for ``<shared>/logs/<machine_id>/...``).
    2. ``<shared>/logs`` — solo / pre-fleet layout SB3 writes today
       (``LOGS_DIR = REPO_ROOT / "logs"`` in :mod:`rl.paths`).
    """
    root = Path(shared_root)
    return [root / "logs" / str(machine_id), root / "logs"]


def latest_explained_variance(
    machine_id: str,
    shared_root: Path,
    *,
    recent_window_seconds: float = DEFAULT_WINDOW_SECONDS,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    aggregator: str = DEFAULT_AGGREGATOR,
    scalar_tag: str = DEFAULT_SCALAR_TAG,
    now_ts: float | None = None,
) -> float | None:
    """Aggregate recent ``train/explained_variance`` for a machine.

    Returns ``None`` when no usable signal exists (no event files, no
    matching tag, all samples outside the window). Callers must
    distinguish ``None`` (no signal) from ``0.0`` (measured zero).
    """
    for logs_dir in _candidate_logs_dirs(str(machine_id), Path(shared_root)):
        files = find_tb_event_files(
            logs_dir, max_age_hours=max_age_hours, now_ts=now_ts
        )
        if not files:
            continue
        samples = scrape_explained_variance(
            files,
            scalar_tag=scalar_tag,
            recent_window_seconds=recent_window_seconds,
            now_ts=now_ts,
        )
        agg = aggregate_recent_ev(samples, aggregator=aggregator)
        if agg is not None:
            return agg
    return None


def _newest_event_file(machine_id: str, shared_root: Path) -> Path | None:
    for logs_dir in _candidate_logs_dirs(str(machine_id), Path(shared_root)):
        files = find_tb_event_files(logs_dir, max_age_hours=DEFAULT_MAX_AGE_HOURS)
        if files:
            return files[0]
    return None


# ---------------------------------------------------------------------------
# CLI sanity probe
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Probe latest train/explained_variance from SB3 TensorBoard events."
    )
    p.add_argument("--machine-id", default="pc-b")
    p.add_argument("--shared-root", type=Path, default=REPO_ROOT)
    p.add_argument("--scalar-tag", default=DEFAULT_SCALAR_TAG)
    p.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    p.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    p.add_argument("--aggregator", default=DEFAULT_AGGREGATOR, choices=list(_VALID_AGGREGATORS))
    args = p.parse_args(argv)

    candidates = _candidate_logs_dirs(args.machine_id, args.shared_root)
    files: list[Path] = []
    chosen_dir: Path | None = None
    for d in candidates:
        files = find_tb_event_files(
            d, max_age_hours=args.max_age_hours
        )
        if files:
            chosen_dir = d
            break
    if not files:
        print(f"[tb_scrape_ev] no event files under any of: {[str(d) for d in candidates]}")
        return 1
    samples = scrape_explained_variance(
        files,
        scalar_tag=args.scalar_tag,
        recent_window_seconds=args.window_seconds,
    )
    agg = aggregate_recent_ev(samples, aggregator=args.aggregator)
    print(
        f"[tb_scrape_ev] dir={chosen_dir} files={len(files)} "
        f"newest={files[0].name} samples={len(samples)} "
        f"aggregator={args.aggregator} value={agg!r}"
    )
    return 0 if agg is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
