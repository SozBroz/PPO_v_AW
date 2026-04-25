# -*- coding: utf-8 -*-
"""Phase 11 Slice D — tests for ``tools/mcts_eval_summary.py``.

Pins the contract used by the orchestrator to build an
:class:`tools.mcts_escalator.EscalatorCycleResult` from on-disk fleet
state. If any of these break, ``run_mcts_escalator`` will quietly emit
``mcts_escalator_no_data`` rows in production.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tools import mcts_eval_summary as _mes
from tools.mcts_baseline import MctsOffBaseline
from tools.mcts_eval_summary import (
    DEFAULT_FALLBACK_SIMS,
    build_cycle_result,
    current_sims_from_proposed,
)


def _baseline(*, wr: float = 0.40) -> MctsOffBaseline:
    return MctsOffBaseline(
        schema_version=1,
        machine_id="m-eval",
        captured_at="2026-04-23T12:30:00Z",
        checkpoint_zip="checkpoints/pool/m-eval/latest.zip",
        checkpoint_zip_sha256="b" * 64,
        games_decided=200,
        winrate_vs_pool=wr,
        mcts_mode="off",
        source="tools/capture_mcts_baseline.py",
    )


def _write_eval(
    fleet_dir: Path, stem: str, *, cw: int, bw: int, mtime: float | None = None
) -> Path:
    eval_dir = fleet_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    p = eval_dir / f"{stem}.json"
    p.write_text(
        json.dumps(
            {
                "candidate_wins": cw,
                "baseline_wins": bw,
                "map_id": 123858,
                "tier": "T3",
                "co_p0": 1,
                "co_p1": 1,
                "per_seat": [],
                "promotion_heuristic_ok": False,
            }
        ),
        encoding="utf-8",
    )
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _write_proposed(
    fleet_dir: Path, *, sims: int | None = None, mode: str = "eval_only"
) -> None:
    fleet_dir.mkdir(parents=True, exist_ok=True)
    args: dict = {"--n-envs": 4, "--mcts-mode": mode}
    if sims is not None:
        args["--mcts-sims"] = int(sims)
    (fleet_dir / "proposed_args.json").write_text(
        json.dumps({"machine_id": fleet_dir.name, "args": args}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# current_sims_from_proposed
# ---------------------------------------------------------------------------


def test_current_sims_missing_file_returns_default(tmp_path: Path) -> None:
    assert current_sims_from_proposed("nope", tmp_path) == DEFAULT_FALLBACK_SIMS


def test_current_sims_reads_int_from_proposed_args(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet" / "m1"
    _write_proposed(fleet, sims=64)
    assert current_sims_from_proposed("m1", tmp_path) == 64


def test_current_sims_returns_default_when_key_missing(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet" / "m1"
    _write_proposed(fleet, sims=None)
    assert current_sims_from_proposed("m1", tmp_path) == DEFAULT_FALLBACK_SIMS


def test_current_sims_returns_default_for_bad_value(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet" / "m1"
    fleet.mkdir(parents=True)
    (fleet / "proposed_args.json").write_text(
        json.dumps({"args": {"--mcts-sims": "not-a-number"}}), encoding="utf-8"
    )
    assert current_sims_from_proposed("m1", tmp_path) == DEFAULT_FALLBACK_SIMS


# ---------------------------------------------------------------------------
# build_cycle_result — happy path
# ---------------------------------------------------------------------------


def test_build_cycle_result_aggregates_verdicts(tmp_path: Path) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=6, bw=4, mtime=time.time() - 30.0)
    _write_eval(fleet, "ckpt_b", cw=5, bw=5, mtime=time.time() - 10.0)

    cycle = build_cycle_result(mid, tmp_path, _baseline(wr=0.40))
    assert cycle is not None
    assert cycle.games_decided == 20
    assert abs(cycle.winrate_vs_pool - (11 / 20)) < 1e-9
    assert cycle.mcts_off_baseline == 0.40
    assert cycle.sims == 16
    assert cycle.engine_desyncs_in_cycle == 0
    assert cycle.explained_variance == 0.0  # TODO scrape


def test_build_cycle_result_returns_none_without_baseline(tmp_path: Path) -> None:
    assert build_cycle_result("m1", tmp_path, None) is None


def test_build_cycle_result_returns_none_without_verdicts(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet" / "m1"
    _write_proposed(fleet, sims=16)
    assert build_cycle_result("m1", tmp_path, _baseline()) is None


def test_build_cycle_result_skips_unparseable_verdicts(tmp_path: Path) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    eval_dir = fleet / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "junk.json").write_text("{not json", encoding="utf-8")
    cycle = build_cycle_result(mid, tmp_path, _baseline())
    assert cycle is None


def test_build_cycle_result_eval_window_caps_inputs(tmp_path: Path) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    base_t = time.time()
    _write_eval(fleet, "old", cw=0, bw=10, mtime=base_t - 100.0)
    _write_eval(fleet, "new", cw=10, bw=0, mtime=base_t - 1.0)

    cycle = build_cycle_result(mid, tmp_path, _baseline(), eval_window=1)
    assert cycle is not None
    assert cycle.games_decided == 10
    assert cycle.winrate_vs_pool == 1.0


# ---------------------------------------------------------------------------
# desync register window proxy
# ---------------------------------------------------------------------------


def test_build_cycle_result_counts_recent_desyncs(tmp_path: Path) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=5, bw=5, mtime=time.time() - 1.0)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "desync_register.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"class": "ok"}),
                json.dumps({"class": "engine_bug", "message": "x"}),
                json.dumps({"class": "oracle_gap", "message": "y"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cycle = build_cycle_result(mid, tmp_path, _baseline(), now_ts=time.time())
    assert cycle is not None
    assert cycle.engine_desyncs_in_cycle == 2


def test_build_cycle_result_ignores_stale_desync_register(tmp_path: Path) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=5, bw=5, mtime=time.time() - 1.0)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    reg = logs / "desync_register.jsonl"
    reg.write_text(json.dumps({"class": "engine_bug"}) + "\n", encoding="utf-8")
    old = time.time() - 10_000.0
    os.utime(reg, (old, old))

    cycle = build_cycle_result(
        mid,
        tmp_path,
        _baseline(),
        cycle_window_seconds=3600.0,
        now_ts=time.time(),
    )
    assert cycle is not None
    assert cycle.engine_desyncs_in_cycle == 0


# ---------------------------------------------------------------------------
# Phase 11d schema_version 2 — per-machine + per-row timestamp filtering
# ---------------------------------------------------------------------------


def _write_register(logs: Path, rows: list[dict]) -> Path:
    logs.mkdir(parents=True, exist_ok=True)
    p = logs / "desync_register.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_count_recent_desyncs_filters_by_machine_id_when_present(
    tmp_path: Path,
) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=5, bw=5, mtime=time.time() - 1.0)
    now = time.time()
    recent_iso = "2026-04-23T12:00:00Z"
    _write_register(
        tmp_path / "logs",
        [
            {
                "schema_version": 2,
                "machine_id": mid,
                "recorded_at": recent_iso,
                "class": "engine_bug",
                "message": "ours",
            },
            {
                "schema_version": 2,
                "machine_id": "other-host",
                "recorded_at": recent_iso,
                "class": "engine_bug",
                "message": "theirs",
            },
            {
                "schema_version": 2,
                "machine_id": mid,
                "recorded_at": recent_iso,
                "class": "ok",
            },
        ],
    )
    # Stamp now_ts at recent_iso so recorded_at falls inside the window.
    parsed_now = _mes._datetime.fromisoformat(
        recent_iso.replace("Z", "+00:00")
    ).timestamp()
    cycle = build_cycle_result(
        mid,
        tmp_path,
        _baseline(),
        cycle_window_seconds=3600.0,
        now_ts=parsed_now + 60.0,
    )
    assert cycle is not None
    # Only the ours/engine_bug row counts; theirs/engine_bug filtered by
    # machine_id, ours/ok filtered by class.
    assert cycle.engine_desyncs_in_cycle == 1
    # Make sure the unused ``now`` is referenced for static analyzers.
    assert now > 0.0


def test_count_recent_desyncs_drops_legacy_rows_under_machine_filter(
    tmp_path: Path,
) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=5, bw=5, mtime=time.time() - 1.0)
    recent_iso = "2026-04-23T12:00:00Z"
    _write_register(
        tmp_path / "logs",
        [
            # Legacy row: no machine_id, no recorded_at. Cannot be
            # attributed; must be ignored when any v2 row exists.
            {"class": "engine_bug", "message": "legacy"},
            # v2 row for our machine.
            {
                "schema_version": 2,
                "machine_id": mid,
                "recorded_at": recent_iso,
                "class": "oracle_gap",
                "message": "ours",
            },
        ],
    )
    parsed_now = _mes._datetime.fromisoformat(
        recent_iso.replace("Z", "+00:00")
    ).timestamp()
    cycle = build_cycle_result(
        mid,
        tmp_path,
        _baseline(),
        cycle_window_seconds=3600.0,
        now_ts=parsed_now + 60.0,
    )
    assert cycle is not None
    assert cycle.engine_desyncs_in_cycle == 1


def test_count_recent_desyncs_recorded_at_excludes_old_rows_even_when_mtime_fresh(
    tmp_path: Path,
) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=5, bw=5, mtime=time.time() - 1.0)
    # File mtime is fresh (the orchestrator just appended), but the
    # specific row was recorded long before the cycle window opened.
    old_iso = "2026-04-22T00:00:00Z"
    new_iso = "2026-04-23T12:30:00Z"
    _write_register(
        tmp_path / "logs",
        [
            {
                "schema_version": 2,
                "machine_id": mid,
                "recorded_at": old_iso,
                "class": "engine_bug",
                "message": "stale",
            },
            {
                "schema_version": 2,
                "machine_id": mid,
                "recorded_at": new_iso,
                "class": "engine_bug",
                "message": "fresh",
            },
        ],
    )
    parsed_now = _mes._datetime.fromisoformat(
        new_iso.replace("Z", "+00:00")
    ).timestamp()
    # 3600s window: old_iso (~36h earlier) is out, new_iso (now) is in.
    cycle = build_cycle_result(
        mid,
        tmp_path,
        _baseline(),
        cycle_window_seconds=3600.0,
        now_ts=parsed_now + 60.0,
    )
    assert cycle is not None
    assert cycle.engine_desyncs_in_cycle == 1


def test_count_recent_desyncs_legacy_register_uses_mtime_fallback(
    tmp_path: Path,
) -> None:
    """When no row carries machine_id or recorded_at, the file-mtime
    gate is the only filter and every non-ok row counts once fresh."""
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=5, bw=5, mtime=time.time() - 1.0)
    _write_register(
        tmp_path / "logs",
        [
            {"class": "ok"},
            {"class": "engine_bug", "message": "legacy1"},
            {"class": "oracle_gap", "message": "legacy2"},
        ],
    )
    cycle = build_cycle_result(
        mid,
        tmp_path,
        _baseline(),
        cycle_window_seconds=3600.0,
        now_ts=time.time(),
    )
    assert cycle is not None
    assert cycle.engine_desyncs_in_cycle == 2


# ---------------------------------------------------------------------------
# Phase 11d EV scrape wiring
# ---------------------------------------------------------------------------


def test_build_cycle_result_uses_scraped_ev_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=6, bw=4, mtime=time.time() - 5.0)

    captured: dict = {}

    def _fake_ev(machine_id, shared_root, *, recent_window_seconds, aggregator, now_ts):
        captured.update(
            machine_id=machine_id,
            shared_root=shared_root,
            recent_window_seconds=recent_window_seconds,
            aggregator=aggregator,
            now_ts=now_ts,
        )
        return 0.83

    monkeypatch.setattr(_mes, "latest_explained_variance", _fake_ev)
    cycle = build_cycle_result(mid, tmp_path, _baseline(wr=0.40))
    assert cycle is not None
    assert cycle.explained_variance == 0.83
    assert captured["machine_id"] == mid
    assert captured["aggregator"] == "median"  # default
    assert captured["recent_window_seconds"] == 3600.0  # default


def test_build_cycle_result_falls_back_to_zero_when_ev_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=5, bw=5, mtime=time.time() - 5.0)

    monkeypatch.setattr(
        _mes, "latest_explained_variance", lambda *a, **kw: None
    )
    cycle = build_cycle_result(mid, tmp_path, _baseline())
    assert cycle is not None
    assert cycle.explained_variance == 0.0


def test_build_cycle_result_passes_ev_aggregator_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = "m-eval"
    fleet = tmp_path / "fleet" / mid
    _write_proposed(fleet, sims=16)
    _write_eval(fleet, "ckpt_a", cw=5, bw=5, mtime=time.time() - 5.0)

    seen: dict = {}

    def _fake_ev(_mid, _root, *, recent_window_seconds, aggregator, now_ts):
        seen["aggregator"] = aggregator
        seen["recent_window_seconds"] = recent_window_seconds
        return 0.71

    monkeypatch.setattr(_mes, "latest_explained_variance", _fake_ev)
    cycle = build_cycle_result(
        mid,
        tmp_path,
        _baseline(),
        ev_aggregator="mean",
        ev_window_seconds=900.0,
    )
    assert cycle is not None
    assert seen["aggregator"] == "mean"
    assert seen["recent_window_seconds"] == 900.0
    assert cycle.explained_variance == 0.71
