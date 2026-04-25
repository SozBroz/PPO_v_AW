# -*- coding: utf-8 -*-
"""Phase 11 Slice E — tests for ``tools/mcts_baseline.py`` + ``tools/capture_mcts_baseline.py``.

These tests pin the on-disk JSON shape that the future Slice D escalator will
consume; if any of them break, the escalator's ``mcts_off_baseline`` field is
also at risk.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import capture_mcts_baseline as cap
from tools.mcts_baseline import (
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_MAX_AGE_HOURS,
    MCTS_OFF_BASELINE_SCHEMA_VERSION,
    MctsOffBaseline,
    baseline_path,
    is_baseline_stale,
    parse_baseline_json,
    read_baseline,
    utc_now_iso_z,
    write_baseline,
)


def _sample(**overrides: Any) -> MctsOffBaseline:
    base = MctsOffBaseline(
        schema_version=MCTS_OFF_BASELINE_SCHEMA_VERSION,
        machine_id="pc-b",
        captured_at="2026-04-23T12:30:00Z",
        checkpoint_zip="checkpoints/pool/pc-b/latest.zip",
        checkpoint_zip_sha256="a" * 64,
        games_decided=200,
        winrate_vs_pool=0.41,
        mcts_mode="off",
        source="tools/capture_mcts_baseline.py",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ---------------------------------------------------------------------------
# write -> read -> parse round-trip
# ---------------------------------------------------------------------------


def test_write_baseline_round_trips_through_read_baseline(tmp_path: Path) -> None:
    shared = tmp_path
    fleet_dir = shared / "fleet" / "pc-b"
    obj = _sample()
    out = write_baseline(obj, fleet_dir)
    assert out == fleet_dir / DEFAULT_BASELINE_FILENAME
    assert out.is_file()

    got = read_baseline("pc-b", shared)
    assert got is not None
    assert got == obj


def test_write_baseline_creates_fleet_dir_recursively(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "deep" / "fleet" / "pc-c"
    assert not fleet_dir.exists()
    out = write_baseline(_sample(machine_id="pc-c"), fleet_dir)
    assert out.parent == fleet_dir
    assert fleet_dir.is_dir()


def test_parse_baseline_json_accepts_round_tripped_dict(tmp_path: Path) -> None:
    obj = _sample()
    out = write_baseline(obj, tmp_path / "fleet" / "pc-b")
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert parse_baseline_json(raw) == obj


def test_parse_baseline_json_returns_none_on_missing_field() -> None:
    raw = {
        "schema_version": MCTS_OFF_BASELINE_SCHEMA_VERSION,
        "machine_id": "pc-b",
        # captured_at missing
        "checkpoint_zip": "x.zip",
        "checkpoint_zip_sha256": "deadbeef",
        "games_decided": 10,
        "winrate_vs_pool": 0.5,
        "mcts_mode": "off",
        "source": "x",
    }
    assert parse_baseline_json(raw) is None


def test_parse_baseline_json_rejects_wrong_schema_version() -> None:
    raw = json.loads(
        json.dumps(
            {
                "schema_version": MCTS_OFF_BASELINE_SCHEMA_VERSION + 1,
                "machine_id": "pc-b",
                "captured_at": "2026-04-23T12:30:00Z",
                "checkpoint_zip": "x.zip",
                "checkpoint_zip_sha256": "abc",
                "games_decided": 10,
                "winrate_vs_pool": 0.5,
                "mcts_mode": "off",
                "source": "x",
            }
        )
    )
    assert parse_baseline_json(raw) is None


def test_parse_baseline_json_rejects_winrate_out_of_range() -> None:
    raw = {
        "schema_version": MCTS_OFF_BASELINE_SCHEMA_VERSION,
        "machine_id": "pc-b",
        "captured_at": "2026-04-23T12:30:00Z",
        "checkpoint_zip": "x.zip",
        "checkpoint_zip_sha256": "abc",
        "games_decided": 10,
        "winrate_vs_pool": 1.5,  # invalid
        "mcts_mode": "off",
        "source": "x",
    }
    assert parse_baseline_json(raw) is None


# ---------------------------------------------------------------------------
# read_baseline error handling
# ---------------------------------------------------------------------------


def test_read_baseline_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert read_baseline("never-existed", tmp_path) is None


def test_read_baseline_returns_none_when_file_malformed(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet" / "pc-b"
    fleet_dir.mkdir(parents=True)
    (fleet_dir / DEFAULT_BASELINE_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_baseline("pc-b", tmp_path) is None


def test_read_baseline_returns_none_when_field_missing(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet" / "pc-b"
    fleet_dir.mkdir(parents=True)
    (fleet_dir / DEFAULT_BASELINE_FILENAME).write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )
    assert read_baseline("pc-b", tmp_path) is None


def test_baseline_path_layout(tmp_path: Path) -> None:
    p = baseline_path("pc-b", tmp_path)
    assert p == tmp_path / "fleet" / "pc-b" / DEFAULT_BASELINE_FILENAME


# ---------------------------------------------------------------------------
# is_baseline_stale boundary
# ---------------------------------------------------------------------------


def _captured_at(now: datetime, age_hours: float) -> str:
    moment = now - timedelta(hours=age_hours)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_is_baseline_stale_at_exact_boundary_is_not_stale() -> None:
    now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
    b = _sample(captured_at=_captured_at(now, DEFAULT_MAX_AGE_HOURS))
    # Exactly one week old → boundary, NOT stale (strict >).
    assert is_baseline_stale(b, max_age_hours=DEFAULT_MAX_AGE_HOURS, now=now) is False


def test_is_baseline_stale_one_second_past_boundary_is_stale() -> None:
    now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
    b = _sample(captured_at=_captured_at(now, DEFAULT_MAX_AGE_HOURS + 1.0 / 3600.0))
    assert is_baseline_stale(b, max_age_hours=DEFAULT_MAX_AGE_HOURS, now=now) is True


def test_is_baseline_stale_one_second_before_boundary_is_fresh() -> None:
    now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
    b = _sample(captured_at=_captured_at(now, DEFAULT_MAX_AGE_HOURS - 1.0 / 3600.0))
    assert is_baseline_stale(b, max_age_hours=DEFAULT_MAX_AGE_HOURS, now=now) is False


def test_is_baseline_stale_unparseable_timestamp_is_stale() -> None:
    b = _sample(captured_at="not-a-timestamp")
    assert is_baseline_stale(b) is True


def test_utc_now_iso_z_format_matches_z_suffix() -> None:
    s = utc_now_iso_z()
    assert s.endswith("Z")
    # Parse back through fromisoformat to assert it round-trips.
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# CLI smoke — patch the eval entrypoint, prove JSON shape
# ---------------------------------------------------------------------------


def _make_fake_checkpoint(path: Path, *, payload: bytes = b"fake-ppo-zip") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _canned_sym_payload(*, cand_wins: int = 41, base_wins: int = 59) -> dict[str, Any]:
    return {
        "candidate_wins": cand_wins,
        "baseline_wins": base_wins,
        "map_id": cap.DEFAULT_MAP_ID,
        "tier": cap.DEFAULT_TIER,
        "co_p0": 1,
        "co_p1": 1,
        "per_seat": {"candidate_as_p0": [21, 50], "candidate_as_p1": [20, 50]},
        "promotion_heuristic_ok": False,
    }


def test_cli_writes_promised_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared = tmp_path / "shared"
    pool_dir = shared / "checkpoints" / "pool" / "pc-b"
    cand_zip = _make_fake_checkpoint(pool_dir / "latest.zip")
    base_zip = _make_fake_checkpoint(shared / "checkpoints" / "promoted" / "best.zip")

    canned = _canned_sym_payload()

    captured_calls: list[dict[str, Any]] = []

    def _fake_eval(**kwargs: Any) -> dict[str, Any]:
        captured_calls.append(kwargs)
        return canned

    monkeypatch.setattr(cap, "_run_symmetric_eval", _fake_eval)

    rc = cap.main(
        [
            "--machine-id",
            "pc-b",
            "--shared-root",
            str(shared),
            "--games",
            "100",
            "--seed",
            "7",
        ]
    )
    assert rc == 0
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["candidate"] == cand_zip.resolve()
    assert call["baseline"] == base_zip.resolve()
    assert call["games_first"] + call["games_second"] == 100
    assert call["seed"] == 7

    out_path = baseline_path("pc-b", shared)
    assert out_path.is_file()
    raw = json.loads(out_path.read_text(encoding="utf-8"))

    # Pin every field the escalator (Slice D) will read.
    assert raw["schema_version"] == 1
    assert raw["machine_id"] == "pc-b"
    assert raw["checkpoint_zip"] == str(cand_zip.resolve())
    assert raw["mcts_mode"] == "off"
    assert raw["source"] == "tools/capture_mcts_baseline.py"
    assert raw["games_decided"] == 100
    assert raw["winrate_vs_pool"] == pytest.approx(0.41)
    assert isinstance(raw["captured_at"], str) and raw["captured_at"].endswith("Z")
    # SHA-256 hex of the fake-ppo-zip bytes.
    import hashlib

    expected = hashlib.sha256(b"fake-ppo-zip").hexdigest()
    assert raw["checkpoint_zip_sha256"] == expected

    # And the helper round-trips it.
    parsed = parse_baseline_json(raw)
    assert parsed is not None
    assert parsed.machine_id == "pc-b"
    assert parsed.mcts_mode == "off"


def test_cli_returns_nonzero_when_candidate_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = tmp_path / "shared"
    (shared / "checkpoints").mkdir(parents=True)

    def _should_not_run(**_: Any) -> dict[str, Any]:
        raise AssertionError("eval entrypoint should not be invoked when candidate missing")

    monkeypatch.setattr(cap, "_run_symmetric_eval", _should_not_run)
    rc = cap.main(["--machine-id", "pc-b", "--shared-root", str(shared)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "candidate checkpoint missing" in err


def test_cli_returns_nonzero_when_eval_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    shared = tmp_path / "shared"
    pool_dir = shared / "checkpoints" / "pool" / "pc-b"
    _make_fake_checkpoint(pool_dir / "latest.zip")
    _make_fake_checkpoint(shared / "checkpoints" / "promoted" / "best.zip")

    def _boom(**_: Any) -> dict[str, Any]:
        raise subprocess.CalledProcessError(returncode=2, cmd=["sym"])

    monkeypatch.setattr(cap, "_run_symmetric_eval", _boom)

    rc = cap.main(
        ["--machine-id", "pc-b", "--shared-root", str(shared), "--games", "10"]
    )
    assert rc == 1
    # No baseline file written when eval failed.
    assert not baseline_path("pc-b", shared).exists()


def test_cli_falls_back_to_latest_zip_when_promoted_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared"
    pool_dir = shared / "checkpoints" / "pool" / "pc-b"
    _make_fake_checkpoint(pool_dir / "latest.zip")
    fallback = _make_fake_checkpoint(shared / "checkpoints" / "latest.zip")

    monkeypatch.setattr(cap, "_run_symmetric_eval", lambda **_: _canned_sym_payload())

    rc = cap.main(["--machine-id", "pc-b", "--shared-root", str(shared), "--games", "20"])
    assert rc == 0

    raw = json.loads(baseline_path("pc-b", shared).read_text(encoding="utf-8"))
    assert raw["mcts_mode"] == "off"
    # We can't introspect the eval kwargs from this test; resolution of the
    # baseline opponent is exercised separately:
    assert cap.resolve_default_opponent(shared) == fallback.resolve()


def test_split_games_first_second_handles_odd_total() -> None:
    assert cap.split_games_first_second(7) == (4, 3)
    assert cap.split_games_first_second(200) == (100, 100)
    assert cap.split_games_first_second(0) == (0, 0)


def test_resolve_pool_latest_falls_back_to_newest_snapshot(tmp_path: Path) -> None:
    shared = tmp_path
    pool_dir = shared / "checkpoints" / "pool" / "pc-b"
    pool_dir.mkdir(parents=True)
    older = pool_dir / "checkpoint_0001.zip"
    newer = pool_dir / "checkpoint_0002.zip"
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    # Bump newer's mtime to be strictly later.
    import os

    os.utime(older, (1.0, 1.0))
    os.utime(newer, (2.0, 2.0))
    assert cap.resolve_pool_latest(shared, "pc-b") == newer
