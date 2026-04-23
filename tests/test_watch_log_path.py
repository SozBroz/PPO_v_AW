"""Tests for the watch-only log path separation.

Phase 10/11 housekeeping (`separate-watch-log-path`): legacy ``log_game`` in
``rl.self_play`` writes the schema-1.5 record for the standalone ``watch_game``
tool. It must land in ``logs/watch_log.jsonl`` — separate from the production
``logs/game_log.jsonl`` that the orchestrator parses on schema >= 1.6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rl.self_play as self_play_mod
from rl.self_play import log_game


def test_log_game_writes_only_to_watch_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watch_path = tmp_path / "watch_log.jsonl"
    game_path = tmp_path / "game_log.jsonl"

    monkeypatch.setattr(self_play_mod, "WATCH_LOG_PATH", watch_path)
    monkeypatch.setattr(self_play_mod, "GAME_LOG_PATH", game_path)

    log_game(
        map_id=123858,
        tier="T1",
        p0_co=1,
        p1_co=1,
        winner=0,
        turns=12,
        funds_end=[1000, 800],
        n_actions=234,
        opening_player=0,
    )

    assert watch_path.is_file(), "log_game must write to watch_log.jsonl"
    assert not game_path.exists(), (
        "log_game must NOT write to game_log.jsonl — that is the production "
        "schema-1.6+ stream consumed by the fleet orchestrator"
    )

    rows = [json.loads(line) for line in watch_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    rec = rows[0]
    assert rec["map_id"] == 123858
    assert rec["log_schema_version"] == "1.5"
    assert rec["winner"] == 0
    assert rec["opening_player"] == 0
    assert "game_id" not in rec, "legacy schema 1.5 deliberately has no game_id"


def test_log_game_appends_multiple_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watch_path = tmp_path / "watch_log.jsonl"
    monkeypatch.setattr(self_play_mod, "WATCH_LOG_PATH", watch_path)

    for w in (0, 1, -1):
        log_game(
            map_id=1,
            tier="T1",
            p0_co=1,
            p1_co=2,
            winner=w,
            turns=5,
            funds_end=[0, 0],
            n_actions=10,
        )

    rows = [json.loads(line) for line in watch_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["winner"] for r in rows] == [0, 1, -1]
