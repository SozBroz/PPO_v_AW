"""tools/slice_game_log.py filter + summary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _row(**kwargs) -> dict:
    base = {
        "turns": 10,
        "winner": 0,
        "map_id": 123858,
        "tier": "T3",
        "log_schema_version": "1.9",
        "learner_seat": 0,
        "reward_mode": "phi",
        "curriculum_tag": "t1",
    }
    base.update(kwargs)
    return base


def test_slice_match_and_summarize() -> None:
    from tools.slice_game_log import _match, _summarize

    assert _match(
        _row(learner_seat=0),
        learner_seat=0,
        curriculum_tag=None,
        reward_mode=None,
        map_id=None,
        tier=None,
        log_schema=None,
        machine_id=None,
        opponent_sampler=None,
        arch_version=None,
    )
    assert not _match(
        _row(learner_seat=1),
        learner_seat=0,
        curriculum_tag=None,
        reward_mode=None,
        map_id=None,
        tier=None,
        log_schema=None,
        machine_id=None,
        opponent_sampler=None,
        arch_version=None,
    )
    assert _match(
        _row(curriculum_tag="x"),
        curriculum_tag="x",
        learner_seat=None,
        reward_mode=None,
        map_id=None,
        tier=None,
        log_schema=None,
        machine_id=None,
        opponent_sampler=None,
        arch_version=None,
    )

    s = _summarize([_row(winner=0, learner_seat=0), _row(winner=1, learner_seat=0)])
    assert s["n"] == 2
    assert s["learner_win_rate"] == 0.5


def test_slice_export_writes_filtered_lines(tmp_path: Path) -> None:
    from tools.slice_game_log import _iter_records, _match

    log = tmp_path / "gl.jsonl"
    log.write_text(
        json.dumps(_row()) + "\n\n" + json.dumps(_row(learner_seat=1, winner=1)) + "\n\n",
        encoding="utf-8",
    )
    rows, bad = _iter_records(log)
    assert bad == 0
    assert len(rows) == 2
    filt = [
        r
        for r in rows
        if _match(
            r,
            curriculum_tag=None,
            learner_seat=1,
            reward_mode=None,
            map_id=None,
            tier=None,
            log_schema=None,
            machine_id=None,
            opponent_sampler=None,
            arch_version=None,
        )
    ]
    out = tmp_path / "out.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in filt:
            fh.write(json.dumps(r) + "\n")
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["learner_seat"] == 1
