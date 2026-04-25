# -*- coding: utf-8 -*-
"""Phase 11d EV scrape — tests for ``tools/tb_scrape_ev.py``.

Strategy: when the ``tensorboard`` runtime can write event files (it
can in this repo's env — see ``requirements.txt`` ``tensorboard>=2.14``)
we write a real ``events.out.tfevents.*`` file and exercise the
``EventAccumulator`` reader end-to-end. If for some reason the writer
import fails, we skip the round-trip cases and still cover the wrapper
logic via monkeypatch.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tools import tb_scrape_ev
from tools.tb_scrape_ev import (
    DEFAULT_SCALAR_TAG,
    ExplainedVarianceSample,
    aggregate_recent_ev,
    find_tb_event_files,
    latest_explained_variance,
    scrape_explained_variance,
)

# ---------------------------------------------------------------------------
# Real TFRecord writer fixture
# ---------------------------------------------------------------------------


def _try_import_writer():
    try:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        return Event, Summary, EventFileWriter
    except Exception:  # noqa: BLE001
        return None


_HAS_WRITER = _try_import_writer() is not None


def _write_events(
    out_dir: Path,
    *,
    samples: list[tuple[float, int, float]],
    tag: str = DEFAULT_SCALAR_TAG,
) -> Path:
    """Write the given (wall_time, step, value) triples into ``out_dir``."""
    Event, Summary, EventFileWriter = _try_import_writer()  # type: ignore[misc]
    out_dir.mkdir(parents=True, exist_ok=True)
    w = EventFileWriter(str(out_dir))
    for wt, step, val in samples:
        s = Summary(value=[Summary.Value(tag=tag, simple_value=float(val))])
        e = Event(wall_time=float(wt), step=int(step), summary=s)
        w.add_event(e)
    w.close()
    files = sorted(out_dir.glob("events.out.tfevents.*"))
    assert files, "writer produced no event files"
    return files[-1]


# ---------------------------------------------------------------------------
# find_tb_event_files
# ---------------------------------------------------------------------------


def test_find_tb_event_files_missing_dir(tmp_path: Path) -> None:
    assert find_tb_event_files(tmp_path / "nope") == []


def test_find_tb_event_files_filters_by_age(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh" / "events.out.tfevents.fresh"
    fresh.parent.mkdir(parents=True)
    fresh.write_bytes(b"x")
    stale = tmp_path / "stale" / "events.out.tfevents.stale"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    very_old = time.time() - 30 * 24 * 3600.0
    os.utime(stale, (very_old, very_old))

    found = find_tb_event_files(tmp_path, max_age_hours=24.0)
    names = [p.name for p in found]
    assert "events.out.tfevents.fresh" in names
    assert "events.out.tfevents.stale" not in names


def test_find_tb_event_files_sorted_newest_first(tmp_path: Path) -> None:
    a = tmp_path / "events.out.tfevents.a"
    b = tmp_path / "events.out.tfevents.b"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    now = time.time()
    os.utime(a, (now - 100.0, now - 100.0))
    os.utime(b, (now - 1.0, now - 1.0))

    found = find_tb_event_files(tmp_path)
    assert found[0].name == "events.out.tfevents.b"
    assert found[1].name == "events.out.tfevents.a"


# ---------------------------------------------------------------------------
# scrape_explained_variance — real TFRecord round trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_WRITER, reason="tensorboard EventFileWriter not importable"
)
def test_scrape_explained_variance_real_roundtrip(tmp_path: Path) -> None:
    now = time.time()
    f = _write_events(
        tmp_path / "logs" / "MaskablePPO_1",
        samples=[(now - 5.0, 0, 0.50), (now - 3.0, 1, 0.60), (now - 1.0, 2, 0.70)],
    )
    samples = scrape_explained_variance(
        [f], scalar_tag=DEFAULT_SCALAR_TAG, recent_window_seconds=3600.0,
        now_ts=now,
    )
    assert len(samples) == 3
    values = sorted(s.value for s in samples)
    assert values == pytest.approx([0.50, 0.60, 0.70], abs=1e-5)
    # Newest-first ordering.
    assert samples[0].step == 2
    assert samples[-1].step == 0


@pytest.mark.skipif(
    not _HAS_WRITER, reason="tensorboard EventFileWriter not importable"
)
def test_scrape_explained_variance_window_excludes_old(tmp_path: Path) -> None:
    now = time.time()
    f = _write_events(
        tmp_path / "logs",
        samples=[
            (now - 7200.0, 0, 0.10),  # 2h old — excluded by 1h window
            (now - 100.0, 1, 0.40),
            (now - 10.0, 2, 0.50),
        ],
    )
    samples = scrape_explained_variance(
        [f], recent_window_seconds=3600.0, now_ts=now
    )
    assert len(samples) == 2
    assert all(s.value > 0.3 for s in samples)


def test_scrape_explained_variance_corrupt_file_skipped(tmp_path: Path) -> None:
    bad = tmp_path / "events.out.tfevents.bad"
    bad.write_bytes(b"garbage-not-a-tfrecord-stream")
    # Must not raise; returns empty.
    samples = scrape_explained_variance(
        [bad], recent_window_seconds=3600.0, now_ts=time.time()
    )
    assert samples == []


def test_scrape_explained_variance_empty_input() -> None:
    assert scrape_explained_variance([], now_ts=time.time()) == []


# ---------------------------------------------------------------------------
# aggregate_recent_ev
# ---------------------------------------------------------------------------


def _mk_sample(wt: float, step: int, val: float) -> ExplainedVarianceSample:
    return ExplainedVarianceSample(wall_time=wt, step=step, value=val)


def test_aggregate_empty_returns_none() -> None:
    assert aggregate_recent_ev([]) is None


def test_aggregate_median() -> None:
    samples = [
        _mk_sample(1.0, 0, 0.10),
        _mk_sample(2.0, 1, 0.50),
        _mk_sample(3.0, 2, 0.90),
    ]
    assert aggregate_recent_ev(samples, aggregator="median") == pytest.approx(0.50)


def test_aggregate_mean() -> None:
    samples = [
        _mk_sample(1.0, 0, 0.10),
        _mk_sample(2.0, 1, 0.50),
        _mk_sample(3.0, 2, 0.90),
    ]
    assert aggregate_recent_ev(samples, aggregator="mean") == pytest.approx(0.50)


def test_aggregate_last_returns_newest_by_walltime() -> None:
    samples = [
        _mk_sample(10.0, 0, 0.20),  # newest
        _mk_sample(5.0, 1, 0.80),
        _mk_sample(1.0, 2, 0.40),
    ]
    assert aggregate_recent_ev(samples, aggregator="last") == pytest.approx(0.20)


def test_aggregate_unknown_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_recent_ev([_mk_sample(1.0, 0, 0.0)], aggregator="bogus")


# ---------------------------------------------------------------------------
# latest_explained_variance — fallback layout
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_WRITER, reason="tensorboard EventFileWriter not importable"
)
def test_latest_explained_variance_falls_back_to_solo_logs(tmp_path: Path) -> None:
    """No <shared>/logs/<mid> dir -> fallback to <shared>/logs."""
    now = time.time()
    _write_events(
        tmp_path / "logs" / "MaskablePPO_1",
        samples=[(now - 30.0, 0, 0.55), (now - 10.0, 1, 0.65)],
    )
    val = latest_explained_variance(
        "pc-b", tmp_path, recent_window_seconds=3600.0, now_ts=now
    )
    assert val is not None
    assert val == pytest.approx(0.60, abs=1e-5)


@pytest.mark.skipif(
    not _HAS_WRITER, reason="tensorboard EventFileWriter not importable"
)
def test_latest_explained_variance_prefers_machine_subdir(tmp_path: Path) -> None:
    """When <shared>/logs/<mid> has events, prefer it over <shared>/logs."""
    now = time.time()
    _write_events(
        tmp_path / "logs" / "MaskablePPO_1",
        samples=[(now - 10.0, 0, 0.10)],  # solo: low EV
    )
    _write_events(
        tmp_path / "logs" / "pc-b" / "MaskablePPO_1",
        samples=[(now - 10.0, 0, 0.90)],  # per-machine: high EV
    )
    val = latest_explained_variance(
        "pc-b", tmp_path, recent_window_seconds=3600.0, now_ts=now
    )
    assert val is not None
    assert val == pytest.approx(0.90, abs=1e-5)


def test_latest_explained_variance_no_logs_returns_none(tmp_path: Path) -> None:
    assert (
        latest_explained_variance(
            "pc-b", tmp_path, recent_window_seconds=3600.0, now_ts=time.time()
        )
        is None
    )


def test_latest_explained_variance_monkeypatch_reader_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the live reader returns no samples, latest_* yields None."""
    # Drop a stub event file so file discovery succeeds.
    f = tmp_path / "logs" / "events.out.tfevents.stub"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    monkeypatch.setattr(
        tb_scrape_ev,
        "_read_one_eventfile_with_accumulator",
        lambda _p, _t: [],
    )
    assert latest_explained_variance("pc-b", tmp_path, now_ts=time.time()) is None


def test_latest_explained_variance_monkeypatch_returns_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wrapper aggregates whatever the reader yields."""
    now = time.time()
    f = tmp_path / "logs" / "events.out.tfevents.stub"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")

    def _fake(_p: Path, _t: str) -> list[ExplainedVarianceSample]:
        return [
            ExplainedVarianceSample(wall_time=now - 5.0, step=0, value=0.42),
            ExplainedVarianceSample(wall_time=now - 1.0, step=1, value=0.58),
        ]

    monkeypatch.setattr(tb_scrape_ev, "_read_one_eventfile_with_accumulator", _fake)
    val = latest_explained_variance(
        "pc-b", tmp_path, recent_window_seconds=3600.0, aggregator="median",
        now_ts=now,
    )
    assert val == pytest.approx(0.50, abs=1e-5)
